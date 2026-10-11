# BLOCK 18b (v4 - C0): Run experiments (VALIDATION only). Finished runs are skipped, so re-running resumes.
# Pick ONE stage, run this cell, then run Block 18c and send back its output.
STAGE = "C0_baseline"

STAGES = {
    # C0: validation baselines for every later ablation (seed 0, eps 8). ~5 runs x ~12-15 min on a T4.
    "C0_baseline": dict(seeds=[0], experiments=[
        ("fedavg_np", None), ("fedprox_np", None), ("dp_clip_nonoise", None),
        ("dp_fixed", 8.0), ("dp_adpmeta", 8.0)]),
    # FINAL (only after all ablations are frozen): 3 paired seeds x every method x eps 1/3/8.
    "final": dict(seeds=[0, 1, 2], experiments=[
        ("fedavg_np", None), ("fedprox_np", None),
        ("dp_fixed", 1.0), ("dp_adpmeta", 1.0),
        ("dp_fixed", 3.0), ("dp_adpmeta", 3.0),
        ("dp_fixed", 8.0), ("dp_adpmeta", 8.0),
        ("fedavg_ch5", None), ("fedprox_ch5", None)]),
}

print(f"Stage {STAGE} | CFG = {CFG}")
run_all(STAGES[STAGE]["experiments"], STAGES[STAGE]["seeds"])
