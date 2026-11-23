# Forecasting Vehicle Fuel Economy Through ML

**Author:** Jake Rah

## Research question

How accurately can combined vehicle fuel economy (MPG) be predicted
from vehicle specifications, and does adding broader vehicle and
technology characteristics improve prediction compared with basic
engine specifications alone?

The approved project compares a training-mean baseline, Linear
Regression, and Decision Tree Regression using two feature sets.

## Interim status — November 21, 2026

Completed:
- Downloaded and inspected the DOE/EPA vehicle dataset.
- Applied the approved conventional gasoline vehicle filters.
- Checked target validity and missing or duplicate vehicle IDs.
- Identified and handled explicit `NA` values in numerical predictors.
- Created a reproducible 80/20 train/test split.
- Evaluated the mean baseline using shuffled five-fold cross-validation
  on the training partition.
- Saved cross-validation results and vehicle split assignments.

Remaining:
- Linear Regression and Decision Tree comparisons for feature sets A and B.
- Decision Tree hyperparameter search and training/validation error analysis.
- Final held-out test evaluation after model selection.
- Model interpretation, diagnostic plots, and largest-error analysis.
- Bundled sample, `main.py` entry point, and clean-environment verification.
- Final `report.md`, including contributions and AI assistance disclosure.

## Data

Source: DOE/EPA FuelEconomy.gov  
Download page: https://www.fueleconomy.gov/feg/download.shtml  
Target: `comb08`, combined MPG.

| Processing stage | Records |
|---|---:|
| Original CSV | 50,647 |
| Gasoline fuel types | 47,532 |
| Empty secondary fuel field | 45,502 |
| Empty alternative-technology field | 43,567 |
| Training partition | 34,853 |
| Held-out test partition | 8,714 |

The original CSV contains 84 columns.

The filters retain Regular, Midgrade, or Premium Gasoline in `fuelType1`,
with empty `fuelType2` and `atvType` fields after whitespace trimming.

The cohort has no target issues detected by the audit and no missing or
duplicate vehicle IDs. Subsequent numerical parsing identified two
missing `displ` values and three missing `cylinders` values encoded as
`NA`. The initial audit's blank counts count empty strings only.

Numerical missing values are imputed using training-fold medians.
Blank technology indicators will be represented as `Not reported`
in the expanded feature pipeline.

Original CSV SHA-256:

```text
e840f1532db12a92a5f57228f137ee08704dd070aa201d02fc5d6d8a5ebe8b74
```

The original `vehicles.csv` is excluded from Git. Preserve this snapshot
locally. The live download may change, so a later download may produce
different counts and results.

## Preliminary baseline results

The baseline is `DummyRegressor(strategy="mean")`. Each fold predicts
the mean target of its own training subset.

The 80/20 split and shuffled five-fold cross-validation use seed 42.
Cross-validation uses only the 34,853-record training partition.
The 8,714-record held-out test partition has not been evaluated.

| Metric | CV mean (MPG) | SD across folds (MPG) |
|---|---:|---:|
| MAE | 3.8156 | 0.0247 |
| RMSE | 4.8706 | 0.0357 |

Lower values are better. Standard deviations use `ddof=0` and describe
variation across the five folds.

Saved evidence:
- [Data audit](results/full/data_audit.json)
- [Split and validation-fold assignments](results/full/split_assignments.csv)
- [Fold-level baseline results](results/full/baseline_cv_folds.csv)
- [Baseline summary](results/full/baseline_cv_summary.csv)

## Run the current baseline

Tested on macOS with Python 3.12.4, NumPy 2.2.6, SciPy 1.15.3,
pandas 2.2.3, and scikit-learn 1.6.1.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the original `vehicles.csv` in the repository root, then run:

```bash
python examine.py --input vehicles.csv --expected-sha256 e840f1532db12a92a5f57228f137ee08704dd070aa201d02fc5d6d8a5ebe8b74
python baseline.py
```

The audit writes `results/full/data_audit.json`.
The baseline writes the split assignments, fold metrics, and summary
listed above. Re-running replaces these generated files.

The current baseline requires the full CSV locally. The offline bundled
sample and required `python main.py` interface are planned next.

## Planned comparisons

- Feature Set A: `displ`, `cylinders`.
- Feature Set B: Set A plus `year`, `VClass`, `drive`, `trany`,
  `fuelType1`, `tCharger`, `sCharger`, and `startStop`.
- Models: mean baseline, Linear Regression, Decision Tree Regression.
- Tree grid: `max_depth` in [3, 5, 8, 12] and
  `min_samples_leaf` in [10, 25, 50].
- Selection metric: training cross-validation MAE, with RMSE also reported.

Identifiers and fuel-economy-derived variables are excluded from model
inputs. Final comparisons will use consistent partitions. Interpretation
will describe predictive associations and the limits of random-split
generalization.

## AI assistance

ChatGPT debugging the `NA` parsing issue, interpreting execution output, and
drafting documentation. The final report will update this disclosure to reflect the
completed work.