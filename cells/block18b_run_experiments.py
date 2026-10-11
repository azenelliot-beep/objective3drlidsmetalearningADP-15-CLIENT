# BLOCK 18b (v4): Run experiments (VALIDATION only). Finished runs are skipped, so re-running resumes.
# Default STAGE = "final": settings FROZEN after Sweeps 1-2 (clip x0.5 + false-alarm penalty 0.5, same reward for
# every method). 3 paired seeds x {FedAvg, FedProx (non-private), fixed-noise DP, ADP-Meta DP with meta-learning}
# x eps {1, 3, 8}. Controllers are meta-trained once per eps on validation and reused for all seeds.
# Then: Block 18c (validation summary), and ONLY THEN set FINAL_TEST = True in Block 19 and run it once.
STAGE = "final"

DP8 = [("dp_fixed", 8.0)]
C0 = [("", {}, [0], [("fedavg_np", None), ("fedprox_np", None), ("dp_clip_nonoise", None),
                     ("dp_fixed", 8.0), ("dp_adpmeta", 8.0)])]
SWEEP1 = [  # (TAG, one change, seeds, runs)
    ("W_val",      dict(WEIGHTS_FROM="val"),                [0], DP8),   # C1 weights from validation counts
    ("B1024_S100", dict(BATCH=1024, LOCAL_STEPS=100),       [0], DP8),   # C3 bigger batches, fewer steps
    ("B1024_S200", dict(BATCH=1024, LOCAL_STEPS=200),       [0], DP8),
    ("B2048_S100", dict(BATCH=2048, LOCAL_STEPS=100),       [0], DP8),
    ("CLIPx0.5",   dict(CLIP_MULT=0.5),                     [0], DP8),   # C3 clip norm
    ("CLIPx2",     dict(CLIP_MULT=2.0),                     [0], DP8),
    ("LR3e-3",     dict(LR=3e-3),                           [0], DP8),   # C3 learning rate
    ("FA0.5",      dict(REWARD_FA=0.5),                     [0], DP8),   # C4 false-alarm penalty
    ("MISS0.5",    dict(REWARD_MISS=0.5),                   [0], DP8),   # C4 missed-attack penalty
    ("POW1.0",     dict(REWARD_POWER=1.0),                  [0], DP8),   # C4 fully balanced weights
    ("MU0",        dict(MU=0.0),                            [0], DP8),   # C11 FedProx mu
    ("MU0.1",      dict(MU=0.1),                            [0], DP8),
]
MAIN = [("",   {},                        [0], [("dp_adpmeta", 8.0)]),     # before
        ("C5", dict(ADP_ACTIONS="pacing"),   [0], [("dp_adpmeta", 8.0)])]     # after: ADP-Meta v2
BASE2 = dict(CLIP_MULT=0.5, REWARD_FA=0.5)                                  # Sweep-1 winners combined
NEXT = [
    ("S2base",    BASE2,                                              [0], [("dp_fixed", 8.0)]),   # reference
    ("S2_ADPv2",  BASE2 | dict(ADP_ACTIONS="pacing"),                 [0], [("dp_adpmeta", 8.0)]), # ADP-Meta online
    ("S2_ADPmeta", BASE2 | dict(ADP_ACTIONS="pacing", ADP_META_EPISODES=10), [0], [("dp_adpmeta", 8.0)]),  # + meta-learning
    ("S2_CLIP0.25", BASE2 | dict(CLIP_MULT=0.25),                     [0], [("dp_fixed", 8.0)]),   # refinements (one change each)
    ("S2_FA1.0",  BASE2 | dict(REWARD_FA=1.0),                        [0], [("dp_fixed", 8.0)]),
    ("S2_MISS0.5", BASE2 | dict(REWARD_MISS=0.5),                     [0], [("dp_fixed", 8.0)]),
    ("S2_B1024",  BASE2 | dict(BATCH=1024, LOCAL_STEPS=200),          [0], [("dp_fixed", 8.0)]),
]
FINAL_CFG = BASE2 | dict(ADP_ACTIONS="pacing", ADP_META_EPISODES=10)        # frozen (CHANGELOG, Sweep 2)
STAGES = {
    "next": NEXT,
    "main": MAIN,
    "auto": C0 + SWEEP1,
    "C0_baseline": C0,
    "sweep1": SWEEP1,
    # ADP-Meta v2 (run after the fixed-noise settings are chosen; add the accepted settings to cfg)
    "C5_pacing_online": [("C5", dict(ADP_ACTIONS="pacing"), [0], [("dp_adpmeta", 8.0)])],
    "C6_meta": [("C6", dict(ADP_ACTIONS="pacing", ADP_META_EPISODES=10), [0], [("dp_adpmeta", 8.0)])],
    # FINAL: frozen settings, 3 paired seeds, eps 1/3/8 (Chapter 5 reference rows are a separate optional stage)
    "final": [("final", FINAL_CFG, [0, 1, 2], [
        ("fedavg_np", None), ("fedprox_np", None),
        ("dp_fixed", 8.0), ("dp_adpmeta", 8.0), ("dp_fixed", 3.0), ("dp_adpmeta", 3.0),
        ("dp_fixed", 1.0), ("dp_adpmeta", 1.0)])],
    "ch5_reference": [("ch5", {}, [0, 1, 2], [("fedavg_ch5", None), ("fedprox_ch5", None)])],
}

for tag, over, seeds, exps in STAGES[STAGE]:
    CFG.clear(); CFG.update(_DEFAULT_CFG); CFG.update(over); CFG["TAG"] = tag
    print(f"\n##### Stage {STAGE} | variant '{tag or 'baseline'}' | changes: {over or 'none'}")
    run_all(exps, seeds)
CFG.clear(); CFG.update(_DEFAULT_CFG)
