# Frozen Importance-Sampling Baseline

Independent template, 2026-11-21. Original template and running jobs unchanged.
Standalone GPU smoke170667 completed three steps, saved checkpoint, with finite
energy/variance/gradient. Step3 total1.827s, sampler0.306s. This validates the
portable entry point, not long-convergence accuracy. Local four CPU tests pass.
Runtime src/ is copied from tested NQS_MPS_seed_check_20261119, not an arbitrary
newer worktree. Implementation scripts are vendored; no experiment directory,
SSH host, checkpoint or absolute workspace dependency is required.

## Run

Use the JAX/CUDA environment validated for RTX4090D (pytorch_new on NQS).
From this directory:

```bash
python run_importance.py --output results/test --steps 8
sbatch run.sbatch
sbatch run.sbatch --tk 0 --steps 3000
```

Defaults: Nphi51, nu1/3, tk17, H397x397 (304103 parameters), D10,
16384 chains, chain_batch16384, forward_batch32768, neighbor tile1024,
burn40, conserved_pair, KFAC lr0.005, damping0.001, clip20, seed512039.
No Fisher projection, no raw-gradient clipping, no PT, no state-row cache.

CPU18-thread fused row-local estimator: exact top32 abs(H) plus32 independent
with-replacement tail draws; q=0.9*abs(H)/sum_tail(abs(H))+0.1/n_tail, with
inverse-probability weights. It is conditionally unbiased, not noiseless.
Only sampled weighted slots are uploaded. Compact GPU dedup and M4 counted
forward remain enabled. No network-dependent value is cached across updates.
Valid-proposal-only MCMC forward is enabled; its startup one-transition gate
checks logprob and acceptance against the unmodified sampler. It may fail
closed on a different numerical backend. Full MCMC convergence is not proven.

## Controls And Records

`--exact` restores full neighbor summation; `--no-mask` restores all-proposal
MCMC forward. Use both for full execution rollback. `--config`, `--tk`,
`--steps`, `--resume` and `--resume-sector` are supported. Resume restores
parameters, NOT chains or optimizer state. Other sizes need new validation.
Do not edit top/draw counts through unsupported environment variables.

Output contains importance_manifest.json, source hashes, input_config.json,
CPU timing JSONL, masked_sampler_validation.json and training/ outputs.
IMPORTANT: original runtime history/resolved may say neighbor_approximation
exact, because they describe the underlying evaluator. Actual estimator is
explicitly recorded in importance_manifest.json; never infer full summation
from that legacy label.

## Evidence And Limits

Job170663 ABBA: fused1.919/1.923s vs prior parallel2.132/2.145s.
Job170665: disabling dedup slower; fixed-input Eloc exactly equal.
Job170666 COMPLETED: masked1.728/1.730s vs original1.948/1.943s, valid
logprob error0 and identical one-transition acceptance in both repetitions.
Short timings are not long-training convergence evidence. Added neighbor
sampling noise must be evaluated separately from MCMC error; periodically use
full-neighbor evaluation for final scientific accuracy checks.

Local CPU checks:
`python -m unittest discover -s implementation -p "test_*.py"`.
