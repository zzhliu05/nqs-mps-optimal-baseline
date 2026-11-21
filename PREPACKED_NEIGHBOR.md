# Experimental prepacked neighbor evaluator

## Additional Experimental Tuning

Set both `execution.neighbor_prepacked=true` and
`execution.neighbor_prepacked_tuned=true` to select eight-row hidden LayerNorm
and two-warps final9->1 contraction, keeping prepare17->9 at four warps.
Both switches default false. Start a fresh process when changing this setting;
execution settings are compilation inputs, not runtime training schedules.

The isolated prototype passed exact common-input full-H locE/gradient checks
and two34-step ABBA rounds showed full-step reductions1.45%/1.68% against the
prepacked untuned path. This small improvement does not meet the3% promotion
gate and is offered for experimental reproduction only. This newly integrated
switch passed its own GPU regression in job170543 (COMPLETED0:0): counts
0,1,1024,1025,32768 were bitwise equal to the prepacked untuned path, cache
clearing passed, and 32768-state medians were18.417->17.916ms and
18.502->17.981ms. Evidence `integrated_tuned_result_170543.json` includes source
hashes. Twelve neighbor tests and three configuration tests also passed. These
short-loop timings do not replace integrated full-training measurements.

Integrated full-training check completed in job170544 (34steps, two warm-up,
two ABBA rounds): untuned/tuned medians2.881606/2.830291s and
3.097548/3.058830s, reductions1.78%/1.25%. Neighbor medians were
4,483,480/4,484,411 and4,806,336/4,827,999 respectively. Evidence
`integrated_combo_abba_170544.json`. This is a small optional improvement, not
a >3% promotion or long-convergence guarantee. Both settings still default off.
No long-run convergence equivalence claim. Raw evidence:
`experiments/nphi51_cuda_20261119/norm8_final2_full_replay.json` and
`combo_abba_summary.json`. Vendor GEMM/CUDA LayerNorm attempts are not enabled.

Opt in with `execution.neighbor_prepacked: true`. Default is false; disable it
to return to the unchanged standard neighbor evaluator. The supported workload
is Nphi=51, backflow H397x397, embedding 128, D=10, M4, reuse_fused_tail,
compact_gpu_dedup, neighbor_skip_unused=true, neighbor_forward_tile=1024.

For the configuration validated here, also set
`execution.mps_scale_stop_gradient: false`. The newer scale-gradient option was
not included in these performance/equality comparisons. Do not conflate the two
changes. Existing training directories and checkpoints are not modified.

Regression: 12 counted-neighbor/tile tests and 2 prepacked configuration tests
passed on CPU JAX. Cache clearing remains supported on the dispatched evaluator.

This changes physical activation stride from 397 to 400 without adding network
parameters or changing the logical LayerNorm width. Hidden/output weights are
padded in a separate JIT call once per evaluation, outside the GPU counted loop.
Packing is recomputed after parameter changes, not cached across training steps.
Sampler and gradient-center evaluators remain standard.

The isolated prototype passed nine-step same-process energy/gradient equality
and KFAC repeatability checks, with about 5.6% lower loss_grad median time.
The integrated module passed nine-step same-process GPU regression on 2026-11-20
(`train_integrated_20261120_140358`): loss, energy, gradients, and repeated KFAC
updates were bitwise equal. This test set mps_scale_stop_gradient=false to isolate
the change. A 34-step integrated run (two warm-up steps, two ABBA rounds with
four-step blocks) measured full-step medians 3.019830 -> 2.898070 seconds and
3.211724 -> 3.108153 seconds, reductions of 4.03% and 3.22%. Packed-mode median
neighbor counts were slightly higher in both rounds, not lower. These are short
trajectory-interleaved measurements, not equal-input full-step replays or long
convergence guarantees. Evidence: experiment `train_integrated_timing_20261120_140714`
and local `experiments/nphi51_cuda_20261119/integrated_abba_summary.json`.
Independent-process
trajectories differed in earlier tests, so this
is not yet a promoted production default. Do not infer convergence equivalence
from the short same-process checks.

No existing configuration is switched on automatically. GPU tests should use
an isolated code snapshot and the same CUDA library environment as production.
