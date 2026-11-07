# Kaur (2024) GRU-assisted FL-DRL IDS: reimplementation on TON_IoT (thesis baseline)

Self-contained. Nothing outside `kaur2024/` is read or written; the Chapter 6 code and results
in the repo root are left untouched.

- `paper/objective2A.pdf`: the paper
- `SPEC.md`: paper verification, PDF-vs-prompt conflicts, all [GAP-FILL] items awaiting approval
- `src/phase1_data.py`: data inspection, split, IID/non-IID partitions, X3 (no training)
- `tools/x4_table5_consistency.py`: X4
- `data/raw/`: put the TON_IoT CSV here (git-ignored)
- `results/`: partitions, reports, extras (`results/extra/`)

Phase 1 data step:

    python kaur2024/src/phase1_data.py --csv kaur2024/data/raw/TON_IoT_Train_Test_Network.csv
