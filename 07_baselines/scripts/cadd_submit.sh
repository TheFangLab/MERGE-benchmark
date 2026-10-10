#!/usr/bin/env bash
# Submit the benchmark VCFs to the public CADD scoring service and collect the
# results. Each VCF is kept under the service's 2 MB / 100,000-variant limit by
# 07_baselines/scripts/make_vcfs.py, so every dataset is one or two submissions.
#
#   bash cadd_submit.sh <vcf-dir> <out-dir>
#
# Writes <out-dir>/<name>.jobid for each submission, then polls until each job
# is finished and downloads <out-dir>/<name>.cadd.tsv.gz.
set -uo pipefail
VCFDIR="${1:-$HOME/merge_baselines/vcf}"
OUT="${2:-$HOME/merge_baselines/cadd}"
BASE=https://cadd.gs.washington.edu
VERSION=GRCh38-v1.7
mkdir -p "$OUT"

submit () {
  local vcf="$1" name; name=$(basename "$vcf" .vcf)
  if [[ -s "$OUT/$name.jobid" ]]; then
    echo "[$name] already submitted: $(cat "$OUT/$name.jobid")"; return
  fi
  local resp; resp=$(curl -s --max-time 300 -X POST "$BASE/upload" \
      -F "file=@$vcf" -F "version=$VERSION")
  local job; job=$(grep -oE 'check_avail/[^"<> ]+\.tsv\.gz' <<<"$resp" | head -1)
  if [[ -z "$job" ]]; then
    echo "[$name] SUBMIT FAILED"; grep -oiE 'error[^<]*' <<<"$resp" | head -3; return 1
  fi
  echo "${job#check_avail/}" > "$OUT/$name.jobid"
  echo "[$name] submitted -> ${job#check_avail/}"
}

collect () {
  local name="$1" job; job=$(cat "$OUT/$name.jobid")
  [[ -s "$OUT/$name.cadd.tsv.gz" ]] && { echo "[$name] already downloaded"; return; }
  local page; page=$(curl -s --max-time 120 "$BASE/check_avail/$job")
  if grep -q 'not yet finished' <<<"$page"; then echo "[$name] pending"; return 1; fi
  # -L matters: the service redirects finished files, and without it curl
  # writes the redirect body and the result looks like a corrupt archive.
  curl -sL --max-time 900 -o "$OUT/$name.cadd.tsv.gz.part" \
       "$BASE/static/finished/$job"
  # Read the header without a pipe: under `pipefail` the SIGPIPE that `head`
  # sends to gzip would make the whole check look like a failure.
  local head3; head3=$(gzip -cd "$OUT/$name.cadd.tsv.gz.part" 2>/dev/null | sed -n '1,3p')
  if grep -q RawScore <<<"$head3"; then
    mv "$OUT/$name.cadd.tsv.gz.part" "$OUT/$name.cadd.tsv.gz"
    echo "[$name] done: $(gzip -cd "$OUT/$name.cadd.tsv.gz" 2>/dev/null | grep -vc '^#') scored rows"
  else
    echo "[$name] download not usable yet, will retry"; return 1
  fi
}

for v in "$VCFDIR"/*.vcf; do submit "$v"; sleep 5; done

echo "--- polling ---"
for round in $(seq 1 240); do
  pending=0
  for j in "$OUT"/*.jobid; do
    n=$(basename "$j" .jobid)
    collect "$n" || pending=$((pending+1))
  done
  [[ $pending -eq 0 ]] && { echo "ALL_CADD_DONE"; break; }
  echo "  $pending pending; sleeping 60s ($(date +%H:%M:%S))"
  sleep 60
done
