#!/usr/bin/env python
"""Score a variant list with big-library NT (old, human-ref 500m) and HyenaDNA.
Same SNV scoring as the genome-wide big-library precompute. Resumable per-model.
Run one model at a time: --model nt | hyena."""
import argparse, csv, os, sys, time
import numpy as np, torch
from pyfaidx import Fasta
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

BASES = ("A", "C", "G", "T")
# Local copies of the published checkpoints were used; set the variables to point
# at yours, or leave them unset to load from the Hugging Face Hub.
NT_PATH = os.environ.get("NT_MODEL_PATH", "InstaDeepAI/nucleotide-transformer-500m-human-ref")
HY_PATH = os.environ.get("HYENADNA_MODEL_PATH", "LongSafari/hyenadna-medium-160k-seqlen-hf")

def norm_chrom(fasta, chrom):
    chrom = str(chrom)
    if chrom in fasta: return chrom
    if ("chr"+chrom) in fasta: return "chr"+chrom
    if chrom.startswith("chr") and chrom[3:] in fasta: return chrom[3:]
    return None

# ---- exact big-library functions ----
def hyena_context(fasta, chrom, pos, context_len=1000):
    start = max(0, pos-1-context_len); end = pos-1
    seq = str(fasta[chrom][start:end]).upper()
    return "".join(b if b in BASES else "N" for b in seq)

def nt_sequences(fasta, chrom, pos, ref, window_size=8192):
    p = pos-1; chr_len = len(fasta[chrom])
    start = max(0, p-window_size//2); end = min(chr_len, p+window_size//2)
    ref_seq = str(fasta[chrom][start:end]).upper()
    offset = p-start
    if offset < 0 or offset >= len(ref_seq) or ref_seq[offset] != ref: return None
    return ref_seq, offset

def mean_pool(last_hidden, mask):
    return (last_hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1, keepdim=True)

def score_nt_batch(model, tok, device, seqs):
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=tok.model_max_length)
    ids = enc["input_ids"].to(device); mask = (ids != tok.pad_token_id).to(device)
    with torch.no_grad():
        outs = model(ids, attention_mask=mask, output_hidden_states=True)
        embs = mean_pool(outs.hidden_states[-1], mask).detach().cpu()
    return embs

def score_hyena_one(model, tok, device, context, ref, alt):
    if not context: return None
    ids = tok(context, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        probs = torch.softmax(model(ids).logits[0, -1, :], dim=0)
    eps = 1e-12
    p_ref = probs[tok.convert_tokens_to_ids(ref)].item()
    p_alt = probs[tok.convert_tokens_to_ids(alt)].item()
    return float(np.log(p_alt+eps) - np.log(p_ref+eps))

def read_variants(path):
    out = []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            out.append((str(r["chr"]), int(r["pos"]), str(r["ref"]).upper(), str(r["alt"]).upper()))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", choices=["nt", "hyena"], required=True)
    ap.add_argument("--fasta", default=os.environ.get("HG38_FASTA", "hg38.fa"))
    ap.add_argument("--nt-batch", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "no cuda"
    fasta = Fasta(args.fasta, sequence_always_upper=True)
    variants = read_variants(args.input)
    total = len(variants)

    done = 0
    if os.path.exists(args.output):
        with open(args.output) as f:
            done = max(0, sum(1 for _ in f) - 1)
    mode = "a" if done > 0 else "w"
    score_col = "nt_mutation_score" if args.model == "nt" else "score"
    fh = open(args.output, mode, newline=""); wr = csv.writer(fh)
    if done == 0:
        wr.writerow(["chr", "pos", "ref", "alt", score_col]); fh.flush()
    print(f"[{args.model}] total={total} resume_from={done}", flush=True)

    if args.model == "nt":
        tok = AutoTokenizer.from_pretrained(NT_PATH)
        model = AutoModelForMaskedLM.from_pretrained(NT_PATH).to(device).eval()
    else:
        tok = AutoTokenizer.from_pretrained(HY_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(HY_PATH, trust_remote_code=True).to(device).eval()

    t0 = time.time()
    i = done
    while i < total:
        if args.model == "nt":
            batch = variants[i:i+args.nt_batch]
            seqs = []; meta = []
            for (c, p, r, a) in batch:
                nc = norm_chrom(fasta, c)
                if nc is None or len(r) != 1 or len(a) != 1:
                    meta.append(("nan", c, p, r, a)); continue
                rs = nt_sequences(fasta, nc, p, r)
                if rs is None:
                    meta.append(("nan", c, p, r, a)); continue
                ref_seq, off = rs
                var_seq = ref_seq[:off] + a + ref_seq[off+1:]
                meta.append((len(seqs), c, p, r, a)); seqs.append(ref_seq); seqs.append(var_seq)
            embs = score_nt_batch(model, tok, device, seqs) if seqs else None
            for m in meta:
                tag, c, p, r, a = m
                if tag == "nan":
                    wr.writerow([c, p, r, a, ""])
                else:
                    sc = float(torch.norm(embs[tag+1] - embs[tag]).item())
                    wr.writerow([c, p, r, a, sc])
            fh.flush(); i += len(batch)
        else:
            c, p, r, a = variants[i]
            nc = norm_chrom(fasta, c)
            if nc is None or len(r) != 1 or len(a) != 1:
                wr.writerow([c, p, r, a, ""])
            else:
                ctx = hyena_context(fasta, nc, p)
                sc = score_hyena_one(model, tok, device, ctx, r, a)
                wr.writerow([c, p, r, a, "" if sc is None else sc])
            fh.flush(); i += 1
        if i % 500 < args.nt_batch:
            el = time.time()-t0; sp = (i-done)/el if el > 0 else 0
            eta = (total-i)/sp if sp > 0 else 0
            print(f"[{args.model}] {i}/{total} {sp:.1f} var/s eta={eta/60:.1f}min", flush=True)
    fh.close()
    print(f"[{args.model}] DONE {total} in {(time.time()-t0)/60:.1f}min", flush=True)

if __name__ == "__main__":
    main()
