import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Merge Enformer summary shard CSVs.")
    parser.add_argument("--shard-output", action="append", required=True)
    parser.add_argument("--final-output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    shards = [Path(path) for path in args.shard_output]
    missing = [str(path) for path in shards if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shard output(s): {missing}")

    frames = [pd.read_csv(path) for path in shards]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("original_index").reset_index(drop=True)

    expected = pd.Series(range(len(merged)), name="original_index")
    if not merged["original_index"].reset_index(drop=True).equals(expected):
        raise ValueError("Merged original_index is not contiguous from 0.")

    final_output = Path(args.final_output)
    merged.to_csv(final_output, index=False)
    print(f"Merged {len(merged)} rows -> {final_output}")


if __name__ == "__main__":
    main()
