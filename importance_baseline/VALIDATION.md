# Validation Scope

Hardware: RTX4090D, single-GPU tests. These are short tests, not scientific
proof of converged energies or sampling equilibrium.

* Full neighbor importance estimator fixed4096-chain check:2000 independent
  draws, added batch-mean RMSE0.009147 vs analytic0.009187; mean real error
  -0.000181. This concerns fixed inputs, not all training distributions.
* Fused CPU ABBA170663:1.919/1.923s vs2.132/2.145s,16384 total chains,
  chain_batch16384,tile1024; first-step compilation excluded.
* GPU dedup ABBA170665: keep dedup1.932/1.948s; bypass1.964/1.975s.
  Fixed-input Eloc identical. Dedup therefore retained.
* Masked sampler ABBA170666:1.728/1.730s vs1.948/1.943s; sampler0.308s vs
  0.518/0.519s. Valid fraction0.63184, max valid logprob error0, identical
  one-transition acceptance. Each new run repeats this startup correctness gate.
* Standalone template170667 completed3 steps and saved params.pkl with finite
  energy, variance and gradient; step3 total1.827s.
* CPU reference checks cover complex coefficients, ties, masks, empty rows,
  short tails and fixed random uniforms. Top64 fork adds a96-slot reference test.

Long runs170656/170664/170673 and top64 run170677 were not complete at freeze;
no final-convergence claim is made. Timings are not portable performance
guarantees. State-row cache was slower in the tested early-training workload
and is deliberately not enabled or required by these entry points.
