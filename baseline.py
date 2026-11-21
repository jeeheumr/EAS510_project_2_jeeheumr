import hashlib
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline

from config import FEATURES_A, N_FOLDS, RAW_CSV, RESULTS, SEED, TARGET, TEST_SIZE
from examine import GASOLINE_TYPES


def load_cohort():
    audit = json.loads((RESULTS / "data_audit.json").read_text())
    checksum = hashlib.sha256(RAW_CSV.read_bytes()).hexdigest()
    if checksum != audit["sha256"] or not audit["cohort_checks_passed"]:
        raise ValueError("Run examine.py on this CSV and check the audit.")

    columns = ["id", "fuelType1", "fuelType2", "atvType", TARGET, *FEATURES_A]
    raw = pd.read_csv(RAW_CSV, usecols=columns, dtype=str, keep_default_na=False)
    eligible = (
        raw["fuelType1"].str.strip().isin(GASOLINE_TYPES)
        & raw["fuelType2"].str.strip().eq("")
        & raw["atvType"].str.strip().eq("")
    )
    cohort = raw.loc[eligible].copy().set_index("id", verify_integrity=True)
    if len(cohort) != audit["filter_counts"]["after_atv_filter"]:
        raise ValueError("Filtered count differs from the audit.")

    # recognize explicit missing values before convert predictor 
    missing_markers = {"", "NA"}

    for column in FEATURES_A:
        values = cohort[column].str.strip()
        missing = values.isin(missing_markers)

        print(
            f"{column}: {int(missing.sum())} missing values "
            "(blank or NA)"
        )

        # Unexpected nonnumeric strings still raise an error.
        cohort[column] = pd.to_numeric(
            values.mask(missing, float("nan")),
            errors="raise",
        )

    # The target must contain valid MPG values.
    cohort[TARGET] = pd.to_numeric(
        cohort[TARGET].str.strip(),
        errors="raise",
    )
    if (
        cohort[TARGET].isna().any()
        or not cohort[TARGET].gt(0).all()
        or cohort[TARGET].isin([float("inf"), float("-inf")]).any()
    ):
        raise ValueError("The target contains invalid MPG values.")

    return cohort


def main():
    cohort = load_cohort()
    train, test = train_test_split(cohort, test_size=TEST_SIZE, random_state=SEED)
    assignments = pd.DataFrame(index=cohort.index)
    assignments["partition"] = "test"
    assignments.loc[train.index, "partition"] = "train"
    assignments["validation_fold"] = pd.Series(index=cohort.index, dtype="Int64")

    pipeline = Pipeline([
        ("preprocess", ColumnTransformer([
            ("engine", SimpleImputer(strategy="median"), FEATURES_A),
        ])),
        ("model", DummyRegressor(strategy="mean")),
    ])

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    records = []
    for fold, (fit_idx, val_idx) in enumerate(folds.split(train), start=1):
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        pipeline.fit(fit[FEATURES_A], fit[TARGET])
        prediction = pipeline.predict(val[FEATURES_A])
        assignments.loc[val.index, "validation_fold"] = fold
        records.append({
            "fold": fold,
            "fit_rows": len(fit),
            "validation_rows": len(val),
            "predicted_mean_mpg": fit[TARGET].mean(),
            "mae_mpg": mean_absolute_error(val[TARGET], prediction),
            "rmse_mpg": root_mean_squared_error(val[TARGET], prediction),
        })

    scores = pd.DataFrame(records)
    metrics = scores[["mae_mpg", "rmse_mpg"]]
    summary = pd.DataFrame({
        "mean": metrics.mean(),
        "std_across_folds": metrics.std(ddof=0),
    })
    assignments.to_csv(RESULTS / "split_assignments.csv")
    scores.to_csv(RESULTS / "baseline_cv_folds.csv", index=False)
    summary.to_csv(RESULTS / "baseline_cv_summary.csv", index_label="metric")

    print(f"Train: {len(train):,} | Held-out test: {len(test):,}")
    print(scores.round(4).to_string(index=False))
    print("\nTraining 5-fold CV, in MPG")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
