import argparse
import os
from pathlib import Path

_preparser = argparse.ArgumentParser(add_help=False)
_preparser.add_argument("--gpu", default="0")
_pre_args, _ = _preparser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)

import joblib
import kipoiseq
import numpy as np
import pandas as pd
import pyfaidx
import tensorflow as tf
import tensorflow_hub as hub
from kipoiseq import Interval
from tqdm import tqdm


SUMMARY_COLUMNS = [
    "enformer_absmean",
    "enformer_mean",
    "enformer_std",
    "enformer_max_abs",
    "enformer_top5_absmean",
    "enformer_pos_sum",
    "enformer_neg_sum",
]


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run Enformer scoring and write seven summary features for one shard."
    )
    parser.add_argument("--gpu", default=str(_pre_args.gpu), help="CUDA device id to expose.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=393216)
    parser.add_argument("--start-index", type=int, required=True, help="Inclusive global row index.")
    parser.add_argument("--end-index", type=int, required=True, help="Exclusive global row index.")
    parser.add_argument("--input-csv", required=True, help="Input variants CSV.")
    parser.add_argument("--output-csv", required=True, help="Shard output CSV.")
    parser.add_argument(
        "--model-path",
        default=str(base_dir / "enformer_model"),
        help="Path to the Enformer model directory.",
    )
    parser.add_argument(
        "--transform-path",
        default=str(base_dir / "genome_data" / "enformer.transform.pkl"),
        help="Path to the preprocessing pipeline.",
    )
    parser.add_argument(
        "--fasta-file",
        default=str(base_dir / "genomes" / "hg38.fa"),
        help="Reference genome FASTA path.",
    )
    return parser.parse_args()


class Enformer:
    def __init__(self, tfhub_url: str):
        self._model = hub.load(tfhub_url).model

    def predict_on_batch(self, inputs):
        predictions = self._model.predict_on_batch(inputs)
        return {key: value.numpy() for key, value in predictions.items()}


class EnformerScoreVariantsRaw:
    def __init__(self, tfhub_url: str, organism: str = "human"):
        self._model = Enformer(tfhub_url)
        self._organism = organism

    def predict_on_batch(self, inputs):
        ref_prediction = self._model.predict_on_batch(inputs["ref"])[self._organism]
        alt_prediction = self._model.predict_on_batch(inputs["alt"])[self._organism]
        return alt_prediction.mean(axis=1) - ref_prediction.mean(axis=1)


class EnformerScoreVariantsNormalized:
    def __init__(self, tfhub_url: str, transform_pkl_path: str, organism: str = "human"):
        self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
        with tf.io.gfile.GFile(transform_pkl_path, "rb") as handle:
            transform_pipeline = joblib.load(handle)
        self._transform = transform_pipeline.steps[0][1]

    def predict_on_batch(self, inputs):
        scores = self._model.predict_on_batch(inputs)
        return self._transform.transform(scores)


class FastaStringExtractor:
    def __init__(self, fasta_file: str):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {key: len(value) for key, value in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(
            interval.chrom,
            max(interval.start, 0),
            min(interval.end, chromosome_length),
        )
        sequence = str(
            self.fasta.get_seq(
                trimmed_interval.chrom,
                trimmed_interval.start + 1,
                trimmed_interval.stop,
            ).seq
        ).upper()
        pad_upstream = "N" * max(-interval.start, 0)
        pad_downstream = "N" * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


def one_hot_encode(sequence: str):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def csv_variant_generator(df: pd.DataFrame):
    for index, row in df.iterrows():
        chrom = str(row["chr"])
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"
        yield kipoiseq.dataclasses.Variant(
            chrom=chrom,
            pos=int(row["pos"]),
            ref=str(row["ref"]).upper(),
            alt=str(row["alt"]).upper(),
            id=f"var_{index}",
        )


def variant_centered_sequences_from_df(df: pd.DataFrame, sequence_length: int, fasta_file: str):
    seq_extractor = kipoiseq.extractors.VariantSeqExtractor(
        reference_sequence=FastaStringExtractor(fasta_file)
    )
    for variant in csv_variant_generator(df):
        interval = Interval(variant.chrom, variant.pos, variant.pos)
        interval = interval.resize(sequence_length)
        center = interval.center() - interval.start
        reference = seq_extractor.extract(interval, [], anchor=center)
        alternate = seq_extractor.extract(interval, [variant], anchor=center)
        yield {
            "inputs": {
                "ref": one_hot_encode(reference),
                "alt": one_hot_encode(alternate),
            },
            "metadata": {
                "original_index": int(variant.id.rsplit("_", 1)[1]),
                "chr": variant.chrom,
                "pos": variant.pos,
                "ref": variant.ref,
                "alt": variant.alt,
            },
        }


def summarize_scores(scores: np.ndarray) -> dict:
    abs_scores = np.abs(scores)
    top_count = min(5, abs_scores.size)
    top_abs = np.partition(abs_scores, -top_count)[-top_count:]
    return {
        "enformer_absmean": float(abs_scores.mean()),
        "enformer_mean": float(scores.mean()),
        "enformer_std": float(scores.std()),
        "enformer_max_abs": float(abs_scores.max()),
        "enformer_top5_absmean": float(top_abs.mean()),
        "enformer_pos_sum": float(scores[scores > 0].sum()),
        "enformer_neg_sum": float(scores[scores < 0].sum()),
    }


def main() -> None:
    args = parse_args()

    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("No GPU detected. Run this on gn1 with CUDA visible.")

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print("Loading Enformer and transform...")
    enformer_scorer = EnformerScoreVariantsNormalized(args.model_path, args.transform_path)

    df_input = pd.read_csv(input_csv)
    total_variants = len(df_input)
    start_index = args.start_index
    end_index = args.end_index

    if start_index < 0 or end_index < 0 or start_index >= end_index:
        raise ValueError("Require 0 <= start-index < end-index.")
    if end_index > total_variants:
        raise ValueError(f"end-index={end_index} exceeds input rows {total_variants}.")

    df_slice = df_input.iloc[start_index:end_index].copy()
    shard_total = len(df_slice)
    print(f"Input rows: {total_variants}; shard [{start_index}, {end_index}) rows: {shard_total}")

    processed_count = 0
    write_mode = "w"
    write_header = True

    if output_csv.exists():
        df_existing = pd.read_csv(output_csv, usecols=["original_index"])
        processed_count = len(df_existing)
        if processed_count > shard_total:
            raise ValueError(f"Existing output has {processed_count} rows but shard has {shard_total}.")
        if processed_count:
            first_idx = int(df_existing["original_index"].iloc[0])
            last_idx = int(df_existing["original_index"].iloc[-1])
            expected_last = start_index + processed_count - 1
            if first_idx != start_index or last_idx != expected_last:
                raise ValueError("Existing output does not match this shard range.")
            print(f"Resuming after {processed_count} completed rows in this shard.")
            write_mode = "a"
            write_header = False

    if processed_count >= shard_total:
        print("Shard already complete.")
        return

    df_to_process = df_slice.iloc[processed_count:].copy()
    iterator = variant_centered_sequences_from_df(
        df_to_process,
        sequence_length=args.sequence_length,
        fasta_file=args.fasta_file,
    )

    batch_refs = []
    batch_alts = []
    batch_metas = []

    progress_label = f"Enformer summary [{start_index}:{end_index}) GPU {args.gpu}"
    with tqdm(total=shard_total, initial=processed_count, desc=progress_label) as pbar:
        for example in iterator:
            batch_refs.append(example["inputs"]["ref"])
            batch_alts.append(example["inputs"]["alt"])
            batch_metas.append(example["metadata"])

            if len(batch_refs) == args.batch_size or (pbar.n + len(batch_refs)) == shard_total:
                batch_inputs = {
                    "ref": tf.stack(batch_refs),
                    "alt": tf.stack(batch_alts),
                }
                variant_scores_batch = enformer_scorer.predict_on_batch(batch_inputs)

                rows_to_write = []
                for batch_idx in range(len(batch_refs)):
                    rows_to_write.append(
                        {**batch_metas[batch_idx], **summarize_scores(variant_scores_batch[batch_idx])}
                    )

                pd.DataFrame(rows_to_write).to_csv(
                    output_csv,
                    columns=["original_index", "chr", "pos", "ref", "alt", *SUMMARY_COLUMNS],
                    mode=write_mode,
                    header=write_header,
                    index=False,
                )

                write_mode = "a"
                write_header = False
                pbar.update(len(batch_refs))
                batch_refs, batch_alts, batch_metas = [], [], []

    print(f"Shard complete: [{start_index}, {end_index}) -> {output_csv}")


if __name__ == "__main__":
    main()
