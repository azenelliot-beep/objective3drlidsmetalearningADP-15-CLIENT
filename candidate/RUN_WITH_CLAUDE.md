# Claude Code handoff

Open chapter6_ddqn_adp_candidate.ipynb. This is a new experimental copy; retain the original unchanged.
The patch is already inserted before the experiment runner. ddqn_private_patch.py is the same patch for review; do not execute it twice.

## Run
1. Create a Python environment with torch, pandas, numpy, scikit-learn, matplotlib, seaborn, imbalanced-learn and opacus. The original first cell pins versions and asserts an environment. Verify available package versions; if those original pins cannot resolve, record actual installed versions and adjust the environment cell only. Never silently skip a dependency error.
2. Attach the ToN-IoT CSV and update DATA_DIR in Block 4. Do not download arbitrary replacement datasets.
3. Execute cells in order with a GPU where available. The stage is repair_check: two settings, each with fixed-noise DP and ADP-Meta, seed 0, epsilon cap 8, delta 1e-5, 15 clients, 20 rounds. This can take hours. Do not reduce clients/rounds or turn off noise for speed.
4. The repaired reference keeps batch 256, 400 steps/round, lr 0.001. Candidate uses batch 512, 200 steps/round, lr 0.0003. Both use identical repaired DDQN, validation-derived rewards, clip multiplier 0.5, false-alarm penalty 0.5, 10 proxy meta episodes and gamma 0.1. Noise is recalibrated for each configuration at the same privacy cap. Candidate superiority is unknown.
5. Compare validation macro-F1, accuracy, FAR, per-class F1, all rounds completed and actual final epsilon. Keep FINAL_TEST=False and CALIB_TEST=False during development. Threshold calibration targets final-tag runs, so it intentionally does not run on these candidate tags.
6. Before long runs: test one private optimizer step, all empty-batch noise handling, target-network influence with gamma>0 vs gamma=0 under matched randomness, positive Gaussian noise, finite gradients and active FedProx. Validate the first-episode policy performance gradient separately from entropy by setting entropy coefficient to zero in the synthetic controller test. Verify frozen policy outputs can vary with state; a constant chosen action does not demonstrate beneficial adaptation. Do not claim these tests passed until executed.
7. If candidate improves validation macro-F1 without unacceptable FAR, freeze the configuration and add seeds 1 and 2 to BOTH repaired reference and candidate. Preserve the same split. Report mean/sample SD and paired ADP-minus-fixed effects. A higher score than an unrepaired gamma-zero baseline is not by itself evidence of ADP benefit.
8. Only after freezing the protocol, enable final test for the frozen tag(s). Do not tune based on test numbers. Save actual stdout, configuration, dependencies, hashes, checkpoints and JSON/CSV metrics. Report failures, early stops and failed FAR caps.

## What changed
- Main per-record private loss now uses an online-selected, target-evaluated nonzero-gamma target.
- Next states are sampled from designated PUBLIC validation half A, independent of private records. This is a synthetic continuing classification MDP, not real sequential flow dynamics. Disclose this design. No private replay buffer is introduced: reusing replay entries requires different privacy analysis.
- Fixed-noise and ADP controls use the same patched loss. FedProx, Gaussian clipping/noise and RDP accounting remain active.
- Corrected zero first-episode reward advantage; policy gradient clipping added.
- Raw local accuracy removed from DPClient telemetry; reward weights use validation counts.
- Disabled FastFederation because its old loss would bypass this repair. Do not re-enable without implementing and testing identical targets/accounting.

## Privacy scope — MUST appear with results
This is conditional record-level DP-SGD research code, NOT an audited end-to-end privacy release. The entire validation set is assumed server-held public proxy data. It is used for proxy training, rewards, anchors, clipping calibration and selection; it is not an independent held-out evaluation. Test is the final evaluation.
Preprocessing and supervised feature selection still fit private training data; label-dependent partitioning, cleaning, sample counts and q are outside the accountant. The default final experiment's private-count reward leak is removed, but these other dependencies remain. To claim end-to-end private raw-record protection, fit transforms/selection on a separately supplied genuinely public dataset, specify fixed public client membership/slots and adjacency, and validate the complete mechanism. Do not relabel private data as public merely to make the claim work. Deterministic seeds are for reproducible benchmarking, not secret noise for production privacy. Multiple model releases/trials require composition analysis.
No achieved performance or formal privacy guarantee is supplied with this package.

## Request to Claude Code
Execute the required synthetic mechanism tests first, repair any runtime errors without changing the stated methods, then run repair_check and report measured results. Do not invent scores, remove privacy noise, include label/type as features, tune on test, or report macro-F1 as accuracy. Ask for the CSV path/GPU only if missing. Keep all main changes concrete and reviewable.
