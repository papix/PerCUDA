# Examples

Every script takes the engine (`gpu` or `perl`; omitted means the GPU when
available) and a problem size as arguments, checks its result against a known
answer and prints `check: ok`. The defaults suit a GPU; pass smaller sizes to
the perl engine. Times below are from a GeForce RTX 4070 SUPER with the
default arguments.

## Research computations

| Script | Field | Method and parameters | Check | GPU time |
| --- | --- | --- | --- | --- |
| `heat.pl` | PDE (parabolic) | 2D heat equation, explicit FTCS, 5-point Laplacian, r = alpha dt / h^2 = 0.24 (stable for r <= 1/4), Dirichlet boundary; 256 x 256, 2000 steps, double precision | exact solution exp(-2 pi^2 t) sin(pi x) sin(pi y); error ~ O(h^2) | 0.11 s |
| `poisson-cg.pl` | PDE (elliptic), linear solvers | -lap(u) = f, 5-point stencil, matrix-free conjugate gradient, relative residual 1e-10; dot products are device reductions; 512 x 512, double precision | manufactured solution u = x (1 - x) sin(pi y); error ~ O(h^2) | 0.14 s (378 iterations) |
| `gray-scott.pl` | Pattern formation | Gray-Scott reaction-diffusion with Karl Sims' parameters (Du = 1, Dv = 0.5, F = 0.055, k = 0.062, dt = 1, 3 x 3 Laplacian -1 / 0.2 / 0.05), periodic; 256 x 256, 10000 steps; optional PGM output | pattern grows and stays bounded | 0.43 s |
| `nbody.pl` | Astrophysics, molecular dynamics | All-pairs gravity, Plummer softening eps = 0.05, G = 1, leapfrog (kick-drift-kick), dt = 0.001, cold collapse of a uniform sphere; 8192 bodies, 1000 steps (6.7e10 interactions) | relative energy drift < 1e-3 (measured 1e-5) | 1.1 s |
| `ising.pl` | Statistical physics, MCMC | 2D Ising model (J = 1), checkerboard Metropolis with precomputed acceptance exp(-4/T), exp(-8/T); 256 x 256, 2000 sweeps at T = 1.5, 2.0, 2.269, 2.5, 3.5; `random($seed, $x, $y, $step)` | Onsager's m(T) below Tc (0.9865 at T = 1.5), ~0 above | 1.8 s |
| `black-scholes.pl` | Computational finance | Black-Scholes closed form with the inputs and A&S 26.2.17 normal CDF of NVIDIA's CUDA samples (S 5..30, X 1..100, T 0.25..10, r = 0.02, sigma = 0.30); 8M options | exact normal CDF on 1000 options, put-call parity on all | 1.2 ms (6.4 Goptions/s) |
| `mc-option.pl` | Monte Carlo methods | European call under GBM, Box-Muller normals, S0 = K = 100, r = 0.05, sigma = 0.2, T = 1; 5e7 paths, double precision; mean and standard error by reductions | within 4 standard errors of 10.4506 | 0.05 s |
| `kmeans.pl` | Machine learning, data analysis | Lloyd's k-means: nearest-centre assignment on the device, centre updates by masked reductions, stop when no label changes; 2M points from 6 Gaussian blobs | found centres match the true ones | 0.28 s |

## Other examples

| Script | Shows |
| --- | --- |
| `matmul.pl` | The example of the PerCUDA paper (Fig. 2) |
| `benchmark.pl` | Matrix multiplication timing with nested arrays |
| `mandelbrot.pl` | Mandelbrot set as a PGM image |
| `pi.pl` | Monte Carlo pi: `random()` and `sum` |

`author/bench-life.pl` (Game of Life with nested arrays, ndarrays and device
arrays) and `author/bench-math.pl` (math functions in every mode) are
benchmarks.

## Patterns used

- Time stepping keeps the state in device arrays (`$runner->array`) and swaps
  them between steps, so nothing is copied until the final `get`.
- Several kernels (one `PerCUDA->new` each) share device arrays.
- Reductions (`sum`, `min`, `max`) give norms, dot products, statistics and
  convergence tests without copying the arrays back.
- `random($seed, ...)` with the thread index and the step number gives
  reproducible random numbers on both engines; Box-Muller turns them into
  normal ones.
- Kernels have no subroutines; a repeated formula can be generated into the
  source by Perl (see the normal CDF in `black-scholes.pl`).

## References

- K. W. Morton and D. F. Mayers, *Numerical Solution of Partial Differential Equations* (FTCS stability).
- J. R. Shewchuk, *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain* (1994).
- J. E. Pearson, "Complex Patterns in a Simple System", *Science* 261 (1993); K. Sims, "Reaction-Diffusion Tutorial".
- L. Nyland, M. Harris and J. Prins, "Fast N-Body Simulation with CUDA", *GPU Gems 3* (2007).
- L. Onsager, *Phys. Rev.* 65, 117 (1944); C. N. Yang, *Phys. Rev.* 85, 808 (1952) (spontaneous magnetization).
- V. Podlozhnyuk, "Black-Scholes Option Pricing", NVIDIA CUDA samples (2007); M. Abramowitz and I. Stegun, 26.2.17.
- P. Glasserman, *Monte Carlo Methods in Financial Engineering* (2003).
- S. Lloyd, "Least squares quantization in PCM", *IEEE Trans. Inf. Theory* 28 (1982).
