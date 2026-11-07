"""X2 [EXTRA]: random top-8 selection (random power per client), IID and non-IID, seeds 0-4.
Isolates what the DQN selection contributes: compared side by side with the DQN (E1/E2) and
FedAvg runs of X1 (same seeds).

  python kaur2024/src/run_x2_random.py [--csv PATH] [--parallel 2] [--tables-only]
Writes kaur2024/results/extra/x2_random.md
"""
import argparse

from run_x1_seeds import SEEDS, tables
from runner_utils import launch

X2 = ["x2_random_iid", "x2_random_noniid"]
COMPARE = ["e1_iid_dqn", "x2_random_iid", "fedavg_iid", "e2_noniid_dqn", "x2_random_noniid", "fedavg_noniid"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args()
    if not a.tables_only:
        launch([(c, s) for s in SEEDS for c in X2], a.parallel, a.csv)
    tables(COMPARE, SEEDS, "X2 [EXTRA] DQN vs random top-8 vs FedAvg (seeds 0-4)",
           "kaur2024/results/extra/x2_random.md")
