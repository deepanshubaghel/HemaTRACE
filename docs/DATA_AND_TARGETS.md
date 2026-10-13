# Data contract and target construction

## Required raw source tables

The raw MIMIC-IV layout is `Dataset/Mimic4/{hosp,icu}/`. The required tables are:

- `hosp/patients.csv` — demographics / anchor age and anchor year.
- `hosp/admissions.csv` — admission metadata/timing.
- `hosp/labevents.csv` — laboratory events including Hb and other curated labs.
- `hosp/d_labitems.csv` — lab `itemid` descriptions/fluids.
- `icu/icustays.csv` — stay IDs, ICU in/out time, LOS.
- `icu/chartevents.csv` — charted vital sign events.
- `icu/d_items.csv` — chart/treatment item metadata.
- `icu/inputevents.csv` — RBC blood product exposures.

The original audit also lists seven optional input tables, which are audited but not required in its existing core execution path. Event extraction is chunked (lab and chart: 2 million rows; ICU input: 1 million).

## Cohort

`base_icu_cohort` merges ICU stays, admissions, and patient demographics. It retains adult stays (`age >= 18`), with at least 24 h of calculated ICU LOS, valid IDs and ICU `intime`. MIMIC-IV anchor ages at/above the source's age threshold are handled by the original age calculation.

Cohort inclusion subsequently requires observed future Hb within `[24, 72)` hours. The original code does not automatically balance classes or impute the outcome.

## Targets and eligibility

| Column | Source code meaning |
| --- | --- |
| `future_min_hb_24_72h` | Smallest valid measured Hb from `[24,72)` h |
| `future_severe_anemia_24_72h` | 1 if minimum future Hb < 8.0 |
| `future_anemia_24_72h` | 1 if Hb < sex-specific threshold |
| `early_min_hb_0_24h` | Minimum Hb in input window |
| `early_anemia_status_0_24h` | Observed baseline anemia status; null if no early Hb |
| `new_onset_anemia_24_72h` | Defined only for observed early-Hb nonanemic stays; otherwise null |
| `future_hb_drop_24_72h` | Last early Hb minus minimum future Hb |

The source thresholds are 13.0 for sex values starting with `M`, and 12.0 for the other recorded category. This choice is retained, including treatment of unknown/other categories; it has **not** been medically reinterpreted in this refactor. The new-onset outcome uses a mask in neural training/evaluation.

## Feature inputs

- Laboratory feature set: hemoglobin, hematocrit, RBC, MCV, MCHC, MCH, RDW, platelets, WBC, creatinine, BUN, lactate, albumin, bilirubin, INR, PT, PTT, ferritin, iron, TIBC, transferrin, B12, folate.
- Vitals: heart rate, systolic/diastolic/mean blood pressure, respiratory rate, SpO₂, temperature, GCS eye/motor/verbal.
- Treatment variables: RBC transfusion event count and amount.
- Temporal inputs are six 4 h bins across the first 24 h, over **35 total temporal feature names**, each with value/mask/delta/count arrays.
- Tabular aggregates include per-stay per-feature count/mean/std/min/max/first/last/slope/missingness where present.

Future measurements are used for outcomes only, not predictor features. The original feature selectors exclude future columns and time-of-discharge/death/total ICU stay information.

## Handling protected health data

Although released MIMIC-IV is de-identified, the data remains controlled-access research data. Keep raw, filtered, and derived patient-level files inside approved storage, and avoid committing `Dataset/`, `HemaTrace_results/`, or checkpoints to GitHub. The `.gitignore` excludes those locations.
