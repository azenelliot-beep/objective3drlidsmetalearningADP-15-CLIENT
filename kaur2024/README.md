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

## Phase 2: running on Kaggle (GPU: "GPU T4 x2"; P100 may not be supported by current PyTorch)

In a Kaggle notebook with the TON_IoT dataset attached:

    !git clone -b claude/inspiring-archimedes-i17mdc https://github.com/azenelliot-beep/objective3drlidsmetalearningADP-15-CLIENT.git repo
    %cd repo
    !python kaur2024/src/phase1_data.py --csv /kaggle/input/datasets/fadiabuzwayed/ton-iot-train-test-network/TON_IoT_Train_Test_Network.csv

1. Smoke test (2 rounds of each core config; checks lr 0.2 and timing):

       !python kaur2024/src/run_core.py --smoke 2 --parallel 4 && tail -n 4 kaur2024/results/logs/*_smoke.log

2. Core runs (seed 0) -> `results/core/core_tables.md`:

       !python kaur2024/src/run_core.py --parallel 4

3. Extras -> `results/extra/x1_seeds.md`, `results/extra/x2_random.md`:

       !python kaur2024/src/run_x1_seeds.py --parallel 4
       !python kaur2024/src/run_x2_random.py --parallel 4

Finished runs are skipped on re-launch. A single run: `python kaur2024/src/fl.py --config kaur2024/configs/e1_iid_dqn.yaml --seed 0`.
Score any model's predictions: `python kaur2024/src/evaluate.py preds.npz [--mask kaur2024/results/partitions/unseen_test_mask.npy]`.
Tests: `python -m pytest kaur2024/tests -q`.
