import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path


GASOLINE_TYPES = {"Regular Gasoline", "Midgrade Gasoline", "Premium Gasoline"}
FEATURES = [
    "displ", "cylinders", "year", "VClass", "drive", "trany",
    "fuelType1", "tCharger", "sCharger", "startStop",
]
REQUIRED = set(FEATURES) | {
    "comb08", "fuelType2", "atvType", "id", "make", "model", "baseModel",
}


def target_issue(value):
    """ usable MPG target must be a finite, positive number."""
    if not value.strip():
        return "missing"
    try:
        number = float(value)
    except ValueError:
        return "not_numeric"
    if not math.isfinite(number):
        return "not_finite"
    if number <= 0:
        return "nonpositive"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("vehicles.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/full/data_audit.json"))
    parser.add_argument("--expected-sha256")
    parser.add_argument("--downloaded-on", type=date.fromisoformat, default=None,
                        help="Actual download date, YYYY-MM-DD, if known.")
    args = parser.parse_args()

    raw = args.input.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if args.expected_sha256 and checksum != args.expected_sha256.lower():
        raise SystemExit("SHA256 mismatch: check the input file before continuing.")

    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    rows = list(reader)
    columns = reader.fieldnames or []
    missing_columns = sorted(REQUIRED - set(columns))
    if missing_columns:
        raise SystemExit(f"Missing required columns: {missing_columns}")
    if not rows:
        raise SystemExit("The CSV has no data rows.")

    # Trim whitespace for comparisons; preserve the original values and bytes.
    gasoline = [r for r in rows if r["fuelType1"].strip() in GASOLINE_TYPES]
    single_fuel = [r for r in gasoline if not r["fuelType2"].strip()]
    cohort = [r for r in single_fuel if not r["atvType"].strip()]

    issues = Counter()
    examples = []
    for row in cohort:
        issue = target_issue(row["comb08"])
        if issue:
            issues[issue] += 1
            if len(examples) < 5:
                examples.append({"id": row["id"], "comb08": row["comb08"],
                                 "issue": issue})

    ids = Counter(r["id"].strip() for r in cohort if r["id"].strip())
    duplicate_ids = sum(count > 1 for count in ids.values())
    missing_ids = sum(not r["id"].strip() for r in cohort)
    predictor_blanks = {
        column: sum(not r[column].strip() for r in cohort) for column in FEATURES
    }
    ready = bool(cohort) and not issues and not duplicate_ids and not missing_ids

    report = {
        "source_url": "https://www.fueleconomy.gov/feg/epadata/vehicles.csv",
        "downloaded_on": args.downloaded_on.isoformat() if args.downloaded_on else None,
        "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_filename": args.input.name,
        "source_bytes": len(raw),
        "sha256": checksum,
        "raw_rows": len(rows),
        "raw_columns": len(columns),
        "filter_counts": {
            "after_gasoline_filter": len(gasoline),
            "after_single_fuel_filter": len(single_fuel),
            "after_atv_filter": len(cohort),
            "excluded_total": len(rows) - len(cohort),
        },
        "filter_rules": {
            "fuelType1_allowed": sorted(GASOLINE_TYPES),
            "fuelType2": "empty after stripping whitespace",
            "atvType": "empty after stripping whitespace",
        },
        "target_issue_counts": dict(issues),
        "invalid_target_examples": examples,
        "duplicate_id_values_in_cohort": duplicate_ids,
        "missing_ids_in_cohort": missing_ids,
        "blank_predictor_counts_in_cohort": predictor_blanks,
        "categorical_blank_policy": "Not reported; never infer confirmed absence",
        "cohort_checks_passed": ready,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("RAW_ROWS:", len(rows))
    for key, value in report["filter_counts"].items():
        print(f"{key.upper()}: {value}")
    print("TARGET_ISSUES:", dict(issues))
    print("DUPLICATE_ID_VALUES:", duplicate_ids)
    print("MISSING_IDS:", missing_ids)
    print("BLANK_PREDICTOR_COUNTS:", predictor_blanks)
    if examples:
        print("INVALID_TARGET_EXAMPLES:", examples)
    print("REPORT:", args.output)
    print("COHORT_CHECKS:", "PASS" if ready else "REVIEW REQUIRED")
    if not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()