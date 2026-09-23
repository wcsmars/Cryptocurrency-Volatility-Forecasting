#!/usr/bin/env python3
"""Independently verify saved real-data labels, losses, provenance and dates.

Every check raises :class:`VerificationError` with a message, so the script
reports failures even when Python is run with ``-O`` (which strips ``assert``).
The recalculations deliberately avoid the package's own feature code: labels
and trailing volatility are rebuilt from the raw Close columns, and aggregate
scores are rebuilt from the per-asset tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION = "Independently verify saved real-data labels, losses, provenance and dates."
MODELS = {"Persistence", "Historical90", "Ridge"}
HORIZON = 30
DATE_COLUMNS = ["origin", "target_start", "target_end", "train_target_end"]


class VerificationError(Exception):
    """A saved result disagrees with an independent recalculation or its provenance."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_directory(metadata: dict, data: Path | None) -> Path:
    if data is not None:
        return Path(data)
    recorded = metadata.get("data_directory")
    if recorded:
        path = Path(recorded)
        return path if path.is_absolute() else ROOT / path
    return ROOT / "data" / "raw"


def _display_path(path: Path) -> str:
    """Repository-relative when possible so summaries never embed a personal path."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _source_close(data: Path, asset: str, input_hashes: dict) -> pd.Series:
    """Load one asset's Close series by the file name recorded for the run.

    Dates are parsed exactly as the loader parses them (ISO 8601, UTC, then
    made naive), so a vendor file with ``T00:00:00Z`` labels verifies too.
    """
    names = [name for name in input_hashes if Path(name).stem == asset]
    _check(len(names) == 1, f"{asset}: expected exactly one recorded source file, found {len(names)}")
    raw = pd.read_csv(data / names[0])
    close = pd.to_numeric(raw["Close"], errors="coerce")
    close.index = pd.DatetimeIndex(pd.to_datetime(raw["Date"], format="ISO8601", utc=True).dt.tz_localize(None))
    return close.sort_index()


def _read_table(path: Path) -> pd.DataFrame:
    """Read a CSV that the pipeline may legitimately have written empty."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _window_volatility(close: pd.Series, first: pd.Timestamp, last: pd.Timestamp) -> float:
    """sqrt(365) times the population standard deviation of the 30 returns in (first, last]."""
    days = pd.date_range(first, last, freq="D")
    values = close.reindex(days).to_numpy(dtype=float)
    if len(values) != HORIZON + 1 or not np.isfinite(values).all() or not (values > 0).all():
        return np.nan
    returns = np.diff(np.log(values))
    return float(np.sqrt(365 * np.mean((returns - returns.mean()) ** 2)))


def verify(output: Path, data: Path | None = None) -> dict:
    output = Path(output)
    metadata = json.loads((output / "run_metadata.json").read_text())
    _check(metadata.get("status") == "complete", "Run did not finish: run_metadata.json status is not 'complete'")
    data = _source_directory(metadata, data)
    _check(data.is_dir(), f"Source directory not found: {data}; run python scripts/download_data.py or pass --data")
    input_hashes = metadata["input_hashes"]
    present = {p.name for p in data.iterdir() if p.is_file() and p.suffix.casefold() == ".csv"}
    _check(present == set(input_hashes),
           "Source directory CSV set differs from the run's inputs: "
           f"unexpected {sorted(present - set(input_hashes))}, missing {sorted(set(input_hashes) - present)}")
    for name, digest in input_hashes.items():
        _check(_sha256(data / name) == digest, f"Source file changed since the run: {name}")
    for name, digest in metadata["code_hashes"].items():
        _check(_sha256(ROOT / name) == digest, f"Code changed since the run: {name}")

    predictions = pd.read_csv(output / "backtest_predictions.csv", parse_dates=DATE_COLUMNS)
    _check(set(predictions.model.unique()) <= MODELS, "backtest_predictions.csv contains an unknown model")
    _check(not predictions.duplicated(["asset", "model", "origin"]).any(), "Duplicate asset/model/origin rows")
    _check(predictions.target_start.eq(predictions.origin + pd.Timedelta(days=1)).all(),
           "target_start is not origin + 1 day for every row")
    _check(predictions.target_end.eq(predictions.origin + pd.Timedelta(days=HORIZON)).all(),
           "target_end is not origin + 30 days for every row")
    fitted = predictions[(predictions.model == "Ridge") & predictions.train_target_end.notna()]
    _check(fitted.train_target_end.le(fitted.origin).all(), "A Ridge fit used a label ending after its forecast origin")
    ok = predictions[predictions.status == "ok"]
    _check(np.isfinite(ok[["actual", "predicted"]]).all().all(), "A successful row has a nonfinite actual or prediction")
    _check(ok[["actual", "predicted"]].ge(0).all().all(), "A successful row has a negative actual or prediction")

    # Recalculate labels directly from source Close (not the feature builder).
    checked = 0
    largest_difference = 0.0
    closes = {}
    for asset, rows in ok.drop_duplicates(["asset", "origin"]).groupby("asset"):
        closes[asset] = _source_close(data, asset, input_hashes)
        chosen = rows.iloc[np.unique(np.linspace(0, len(rows) - 1, min(5, len(rows))).astype(int))]
        for row in chosen.itertuples():
            direct = _window_volatility(closes[asset], row.origin, row.target_end)
            _check(np.isfinite(direct), f"{asset} {row.origin.date()}: target window lacks 31 consecutive valid closes")
            difference = abs(direct - row.actual)
            largest_difference = max(largest_difference, difference)
            _check(np.isclose(direct, row.actual, rtol=1e-8, atol=1e-10),
                   f"{asset} {row.origin.date()}: saved actual {row.actual} differs from recalculated {direct}")
            checked += 1

    # Independent matched-date losses, including a fixed nonoverlap schedule.
    metrics = pd.read_csv(output / "backtest_metrics.csv")
    metric_checks = 0
    for asset, rows in predictions.groupby("asset"):
        complete = rows[rows.status == "ok"].groupby("origin").model.nunique()
        common = complete.index[complete == len(MODELS)]
        calendar = pd.date_range(rows.origin.min(), rows.origin.max(), freq="D")
        for sample, requested in [("daily_origins", calendar), ("nonoverlap_30d", calendar[::HORIZON])]:
            scored_dates = common.intersection(requested)
            for model in MODELS:
                label = f"{asset}/{model}/{sample}"
                selected = metrics[(metrics.asset == asset) & (metrics.model == model) & (metrics["sample"] == sample)]
                _check(len(selected) == 1, f"{label}: expected exactly one metric row")
                observed = selected.iloc[0]
                values = rows[(rows.model == model) & rows.origin.isin(scored_dates)]
                error = values.predicted.to_numpy() - values.actual.to_numpy()
                _check(observed.n_requested == len(requested), f"{label}: n_requested differs")
                _check(observed.n_scored == len(error), f"{label}: n_scored differs")
                _check(np.isclose(observed.coverage, len(error) / len(requested)), f"{label}: coverage differs")
                if len(error):
                    _check(np.isclose(observed.mae, np.mean(np.abs(error))), f"{label}: MAE differs")
                    _check(np.isclose(observed.rmse, np.sqrt(np.mean(error ** 2))), f"{label}: RMSE differs")
                else:
                    _check(pd.isna(observed.mae) and pd.isna(observed.rmse), f"{label}: unscored sample has a loss")
                metric_checks += 1

    # Aggregate scores are equal-weight means over assets scored for all models.
    # The table is legitimately empty when no asset is scored for every model.
    macro = _read_table(output / "macro_metrics.csv")
    expected_rows = set()
    for sample in metrics["sample"].unique() if len(metrics) else []:
        subset = metrics[metrics["sample"] == sample]
        valid = subset[subset.n_scored.gt(0) & subset.mae.notna() & subset.rmse.notna()]
        if valid.groupby("asset").model.nunique().eq(len(MODELS)).any():
            expected_rows |= {(sample, model) for model in MODELS}
    observed_rows = set(zip(macro["sample"], macro["model"])) if len(macro) else set()
    _check(observed_rows == expected_rows,
           f"macro_metrics.csv rows {sorted(observed_rows)} differ from the per-asset table {sorted(expected_rows)}")
    macro_checks = 0
    for row in macro.itertuples():
        subset = metrics[metrics["sample"] == row.sample]
        valid = subset[subset.n_scored.gt(0) & subset.mae.notna() & subset.rmse.notna()]
        counts = valid.groupby("asset").model.nunique()
        group = valid[valid.asset.isin(counts.index[counts == len(MODELS)]) & valid.model.eq(row.model)]
        label = f"macro {row.sample}/{row.model}"
        _check(row.n_assets == len(group), f"{label}: n_assets differs")
        _check(row.n_scored == int(group.n_scored.sum()), f"{label}: n_scored differs")
        _check(np.isclose(row.macro_mae, group.mae.mean()), f"{label}: macro MAE differs")
        _check(np.isclose(row.macro_rmse, group.rmse.mean()), f"{label}: macro RMSE differs")
        macro_checks += 1

    # Latest forecasts must be dated at each asset's last source date, and the
    # persistence value must equal the trailing 30-day volatility of raw closes.
    latest = pd.read_csv(output / "next_forecasts.csv", parse_dates=DATE_COLUMNS)
    latest_checks = 0
    for row in latest.itertuples():
        if row.asset not in closes:
            closes[row.asset] = _source_close(data, row.asset, input_hashes)
        close = closes[row.asset]
        _check(row.origin == close.index.max(), f"{row.asset}: latest forecast origin is not the last source date")
        _check(row.target_end == row.origin + pd.Timedelta(days=HORIZON), f"{row.asset}: latest target_end is wrong")
        if row.model == "Persistence" and row.status == "ok":
            direct = _window_volatility(close, row.origin - pd.Timedelta(days=HORIZON), row.origin)
            _check(np.isclose(direct, row.predicted, rtol=1e-8, atol=1e-10),
                   f"{row.asset}: latest persistence forecast differs from trailing volatility of raw closes")
            latest_checks += 1

    daily = metrics[metrics["sample"] == "daily_origins"]
    _check(metadata["scored_prediction_rows"] == int(daily.n_scored.sum()),
           "run_metadata.json scored_prediction_rows disagrees with backtest_metrics.csv")
    return {
        "status": "passed",
        "source_directory": _display_path(data),
        "source_files_verified": len(input_hashes),
        "code_files_verified": len(metadata["code_hashes"]),
        "source_windows_independently_recalculated": checked,
        "max_absolute_target_difference": largest_difference,
        "metric_rows_independently_recalculated": metric_checks,
        "macro_rows_independently_recalculated": macro_checks,
        "latest_persistence_forecasts_recalculated": latest_checks,
        "all_forecast_dates_and_training_cutoffs_validated": True,
        "scored_prediction_rows": int(daily.n_scored.sum()),
        "forecast_origin_observations_per_model": int(daily[daily.model == "Ridge"].n_scored.sum()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("output", type=Path, help="Result directory written by python -m crypto_volatility")
    parser.add_argument("--data", type=Path, default=None,
                        help="Source CSV directory (default: the data_directory recorded in run_metadata.json)")
    args = parser.parse_args(argv)
    try:
        summary = verify(args.output, args.data)
    except (VerificationError, OSError, KeyError, ValueError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
