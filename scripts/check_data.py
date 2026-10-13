"""Check only the raw MIMIC-IV input tree; do not launch model training."""

import argparse
from pathlib import Path

REQUIRED = (
    "hosp/patients.csv",
    "hosp/admissions.csv",
    "hosp/labevents.csv",
    "hosp/d_labitems.csv",
    "icu/icustays.csv",
    "icu/chartevents.csv",
    "icu/d_items.csv",
    "icu/inputevents.csv",
)
OPTIONAL = (
    "hosp/diagnoses_icd.csv",
    "hosp/d_icd_diagnoses.csv",
    "hosp/prescriptions.csv",
    "hosp/pharmacy.csv",
    "hosp/emar.csv",
    "hosp/emar_detail.csv",
    "icu/procedureevents.csv",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate HemaTRACE input filenames")
    parser.add_argument("--root", type=Path, default=Path("Dataset/Mimic4"))
    args = parser.parse_args()
    missing = []
    for name in (*REQUIRED, *OPTIONAL):
        p = args.root / name
        required = name in REQUIRED
        state = "OK" if p.is_file() else "MISSING"
        print(f"[{state:7}] {'required' if required else 'optional':8} {p}")
        if required and not p.is_file():
            missing.append(str(p))
    if missing:
        print(f"\nMissing {len(missing)} required file(s).")
        return 1
    print("\nAll required files are present. This does not validate CSV schemas/quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
