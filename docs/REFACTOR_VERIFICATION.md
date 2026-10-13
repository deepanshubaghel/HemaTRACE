# Refactor verification and preservation scope

## Original source

`HemaTraceF(20261013-050659).py` — SHA-256 in `SOURCE_SHA256.txt`.

## What was preserved

The original single script contains 178 class/function definition occurrences (174 unique names), counting four later redefinitions. An AST structural comparison of every `FunctionDef`, `AsyncFunctionDef`, and `ClassDef` (including decorators, default values, signatures, methods, bodies, and expression trees) against all source files of this package returned:

```text
Original distinct function/class names: 174
Refactored distinct function/class names: 174
Original function/class definition occurrences: 178
Refactored function/class definition occurrences: 178
Missing names: 0
Mismatched AST definitions: 0
```

Original module-level config default assignments, curated itemid variable lists, complexity symbols, and ablation/module comparison overrides were retained; only imports/namespace boundaries, command entry points, dependencies, docs and synthetic tests were reorganized or added.

### Not an equivalence proof

This AST assertion verifies **function/class code structure**, not empirical numerical equivalence. New explicit inter-module imports and optional dependency imports are necessary because Python functions no longer all share one monolithic global namespace. The later comparison override aliases are preserved within the same comparison module and exposed to the pipeline. The original data and trained model outputs are unavailable here, so a real full-run parity test has **not** been performed. Existing code issues, if any, were intentionally not silently modified.

## Suggested end-to-end verification with authorized data

1. Record MIMIC-IV version, source SHA, hardware and dependency versions.
2. Run original script and modular implementation in equivalent isolated environments on the same authorized data.
3. Compare cohort counts, item ID maps, split assignments and temporal tensor shapes.
4. Compare predictions and metrics within numerical tolerances appropriate to stochastic GPU training.
5. Review all warnings and missing output stages, and document any intentional changes in a new version.

## Why it is not a single script anymore

Every original function/class definition is located under a dedicated package responsibility, with a short `main.py` launcher and central `hematrace.pipeline.main()` orchestrator. The DL comparator file remains relatively large because its many related benchmark architectures and later self-overriding helper implementations need to share module globals to retain original behavior.
