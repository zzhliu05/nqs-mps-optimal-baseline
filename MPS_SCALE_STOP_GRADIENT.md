# MPS scale autodiff

Contraction rescaling now defaults to stop-gradient. Division and logarithmic
compensation use the same detached scale. Forward values, Frobenius scaling,
scale floors, amplitude clipping and normalization frequency are unchanged.
This does not remove the wavefunction's global amplitude direction.

The change covers differentiable sequential/tree/bank contractions and the
standard-JAX VJP references for fused M3/M4, including normalized tree tails.
Forward-only Pallas kernels require no change to their numerical operations.
This is not a conversion of the wavefunction interface to sign/log-amplitude.

Rollback with `execution.mps_scale_stop_gradient: false` in the JSON config.
Direct legacy entrypoints can use `NQS_MPS_SCALE_STOP_GRADIENT=0`.
Switch before JIT tracing, or start a fresh process.

FP32 gradients away from nodes and clipping should agree up to numerical error.
BF16 casts can break exact scale cancellation; training equivalence is not
claimed from the contraction identity alone. Existing remote jobs are unchanged.
