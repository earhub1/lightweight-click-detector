"""Utilities to convert detector CSV outputs into a single Raven table."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd


DEFAULT_LOW_FREQ = 1_000
DEFAULT_HIGH_FREQ = 180_000
RAVEN_COLUMNS = [
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Begin File",
    "File Offset (s)",
]


@dataclass
class RavenRow:
    begin_time: float
    end_time: float
    low_freq: float
    high_freq: float
    begin_file: str
    file_offset: float

    def to_dict(self) -> dict:
        return {
            "Begin Time (s)": self.begin_time,
            "End Time (s)": self.end_time,
            "Low Freq (Hz)": self.low_freq,
            "High Freq (Hz)": self.high_freq,
            "Begin File": self.begin_file,
            "File Offset (s)": self.file_offset,
        }


def infer_frame_duration(time_series: pd.Series) -> float:
    """Infer the temporal resolution (frame duration) from time stamps."""

    if time_series.size <= 1:
        return 0.001  # falls back to default short-term duration used by the detector

    diffs = time_series.diff().dropna()
    # Guard against degenerate cases (e.g. repeated timestamps)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.empty:
        return 0.001
    return float(positive_diffs.median())


def csv_to_raven_rows(
    csv_path: Path,
    base_dir: Path,
    cumulative_offset: float,
    low_freq: float,
    high_freq: float,
) -> tuple[List[RavenRow], float]:
    """Convert a detector CSV to Raven rows and return the new cumulative offset."""

    df = pd.read_csv(csv_path)
    required_columns = {"time_sec", "high_click_events", "low_click_events"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"File '{csv_path}' is missing required columns: {', '.join(sorted(missing))}"
        )

    df = df.sort_values("time_sec").reset_index(drop=True)

    frame_duration = infer_frame_duration(df["time_sec"])

    # Keep rows with at least one detected event
    mask = ~((df["high_click_events"] == 0) & (df["low_click_events"] == 0))
    df_events = df.loc[mask].copy()

    raven_rows: List[RavenRow] = []
    if not df_events.empty:
        # Convert timestamps from frame centres to frame start/end.
        start_times_local = (df_events["time_sec"] - frame_duration / 2).clip(lower=0.0)
        end_times_local = start_times_local + frame_duration

        begin_times_global = cumulative_offset + start_times_local
        end_times_global = cumulative_offset + end_times_local

        begin_file = csv_path.relative_to(base_dir).with_suffix(".wav").as_posix()

        for begin_time, end_time, offset in zip(
            begin_times_global,
            end_times_global,
            start_times_local,
        ):
            raven_rows.append(
                RavenRow(
                    begin_time=float(begin_time),
                    end_time=float(end_time),
                    low_freq=float(low_freq),
                    high_freq=float(high_freq),
                    begin_file=begin_file,
                    file_offset=float(offset),
                )
            )

    # Update the cumulative offset using the full extent of the file (not only detected rows)
    max_time = float(df["time_sec"].max()) if not df.empty else 0.0
    file_duration = max_time + frame_duration / 2
    new_offset = cumulative_offset + file_duration

    return raven_rows, new_offset


def gather_csv_files(input_dir: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(input_dir.glob(pattern))


def convert_directory(
    input_dir: Path,
    output_file: Path,
    recursive: bool,
    low_freq: float,
    high_freq: float,
) -> None:
    input_dir = input_dir.resolve()
    csv_files = list(gather_csv_files(input_dir, recursive))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{input_dir}'.")

    all_rows: List[RavenRow] = []
    cumulative_offset = 0.0

    for csv_path in csv_files:
        rows, cumulative_offset = csv_to_raven_rows(
            csv_path,
            base_dir=input_dir,
            cumulative_offset=cumulative_offset,
            low_freq=low_freq,
            high_freq=high_freq,
        )
        all_rows.extend(rows)

    raven_df = pd.DataFrame([row.to_dict() for row in all_rows], columns=RAVEN_COLUMNS)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    raven_df.to_csv(output_file, sep="\t", index=False, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert detector outputs (CSV files) into a single Raven-compatible table."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the CSV detection files.",
    )
    parser.add_argument(
        "output_txt",
        type=Path,
        help="Path to the consolidated Raven tab-delimited text output.",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only consider CSV files in the top-level of input_dir.",
    )
    parser.add_argument(
        "--low-freq",
        type=float,
        default=DEFAULT_LOW_FREQ,
        help="Low frequency boundary in Hz (default: %(default)s).",
    )
    parser.add_argument(
        "--high-freq",
        type=float,
        default=DEFAULT_HIGH_FREQ,
        help="High frequency boundary in Hz (default: %(default)s).",
    )
    parser.set_defaults(recursive=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_directory(
        input_dir=args.input_dir,
        output_file=args.output_txt,
        recursive=args.recursive,
        low_freq=args.low_freq,
        high_freq=args.high_freq,
    )


if __name__ == "__main__":
    main()
