# BLOCK 18b (v4): Run experiments (VALIDATION only). Finished runs are skipped, so re-running resumes.
# Pick ONE stage, run this cell, then run Block 18c and send back its output.
# Each stage applies its CFG overrides on top of the defaults in Block 18a; every ablation has its own TAG,
# so results of different configs can never be mixed up (Block 18a refuses a config change without a new TAG).
STAGE = "C0_baseline"

STAGES = {
    # C0: validation baselines for every later ablation (seed 0, eps 8). ~5 runs x ~12-15 min on a T4.
    "C0_baseline": dict(cfg={}, seeds=[0], experiments=[
        ("fedavg_np", None), ("fedprox_np", None), ("dp_clip_nonoise", None),
        ("dp_fixed", 8.0), ("dp_adpmeta", 8.0)]),
    # C5: ADP-Meta v2 (RDP budget pacing, macro-F1 reward, 32-16 policy), learning online in the real run.
    "C5_pacing_online": dict(cfg=dict(ADP_ACTIONS="pacing", TAG="C5"), seeds=[0], experiments=[
        ("dp_adpmeta", 8.0)]),
    # C6: + meta-training on validation (half A train / half B reward), frozen greedy policy in the real run.
    #     Each practice episode costs about one real run (~15 min): 10 episodes ~ 2.5 h per eps budget.
    "C6_meta": dict(cfg=dict(ADP_ACTIONS="pacing", ADP_META_EPISODES=10, TAG="C6"), seeds=[0], experiments=[
        ("dp_adpmeta", 8.0)]),
    # FINAL (only after all ablations are frozen): 3 paired seeds x every method x eps 1/3/8.
    # Put the accepted ablation settings into cfg (and a TAG) before running.
    "final": dict(cfg={}, seeds=[0, 1, 2], experiments=[
        ("fedavg_np", None), ("fedprox_np", None),
        ("dp_fixed", 1.0), ("dp_adpmeta", 1.0),
        ("dp_fixed", 3.0), ("dp_adpmeta", 3.0),
        ("dp_fixed", 8.0), ("dp_adpmeta", 8.0),
        ("fedavg_ch5", None), ("fedprox_ch5", None)]),
}

CFG.clear(); CFG.update(_DEFAULT_CFG); CFG.update(STAGES[STAGE]["cfg"])
print(f"Stage {STAGE} | CFG = {CFG}")
run_all(STAGES[STAGE]["experiments"], STAGES[STAGE]["seeds"])
