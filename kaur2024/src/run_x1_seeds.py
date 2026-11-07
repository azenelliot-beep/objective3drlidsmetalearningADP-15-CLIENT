"""X1 [EXTRA]: training seeds 0-4 for E1, E2 and FedAvg (IID and non-IID); mean ± std (sample std,
ddof=1) at rounds 50 and 100. Seed 0 is the core run (results/core); seeds 1-4 go to
results/extra/runs. Partitions are fixed (G9): only the training seed changes.

  python kaur2024/src/run_x1_seeds.py [--csv PATH] [--parallel 2] [--tables-only]
Writes kaur2024/results/extra/x1_seeds.md
"""
import argparse

from run_core import CORE
from runner_utils import CLASS_NAMES, launch, mean_std_table, repo_path

SEEDS = [0, 1, 2, 3, 4]
KEYS = ["acc", "macro_P", "macro_R", "macro_F1", "AUC", "best_client", "worst_client",
        "unseen_acc", "unseen_macro_F1", "energy_cum_J"]


def tables(configs, seeds, title, path):
    L = [f"# {title}", "", "Mean ± sample std over training seeds (partition fixed, seed 0). "
         "Percent, except energy (cumulative C_training, J).", "",
         "## Summary metrics", ""]
    L += mean_std_table(configs, seeds, (1, 50, 100), KEYS)
    L += ["", "## Per-class F1", ""]
    L += mean_std_table(configs, seeds, (50, 100), [f"{c}_F1" for c in CLASS_NAMES], CLASS_NAMES)
    p = repo_path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args()
    if not a.tables_only:
        launch([(c, s) for s in SEEDS for c in CORE], a.parallel, a.csv)
    tables(CORE, SEEDS, "X1 [EXTRA] training seeds 0-4", "kaur2024/results/extra/x1_seeds.md")
