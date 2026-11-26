# Contributing to PerCUDA

## Setup

- Perl 5.40 or later (CI covers 5.40, 5.42 and 5.44).
- For the GPU engine: [CUDA::DriverAPI](https://github.com/papix/CUDA-DriverAPI)
  with NVRTC, and an NVIDIA GPU. Without them, everything runs on the perl
  engine and the GPU tests skip. It is not on CPAN yet; install it from its
  repository with `minil install`.
- Dependencies, including the author tools and the optional PDL (needed for
  `t/08_pdl.t`; `t/09_without_pdl.t` checks PerCUDA also works without it):

  ```sh
  cpanm --with-configure --with-develop --installdeps .
  ```

## Build and test

```sh
prove -l t/                    # unit tests; GPU tests skip without a device
minil test --release           # plus POD, spelling, Perl::Critic, examples
perl eg/benchmark.pl gpu 1024  # matrix multiplication, one run
perl author/bench-life.pl      # Game of Life: perl engine, nested, ndarray and device arrays
perl author/bench-math.pl      # math functions in every mode (fast, compat, exact)
```

Compare a change with the previous commit by running the same benchmarks on
both, alternating, and run the perl engine last: its large allocations slow
down whatever is measured after it in the same process.

Run the full suite on a machine with an NVIDIA GPU before sending a change:
the tests that compare the perl engine with the GPU only run there.

## Lint, formatting and secrets

```sh
author/lint.sh         # perltidy, perlcritic, English only, actionlint, gitleaks (tree and history)
author/lint.sh --fix   # reformat Perl files with .perltidyrc first
```

It needs `perltidy` and `perlcritic` (develop dependencies), plus
[gitleaks](https://github.com/gitleaks/gitleaks) and
[actionlint](https://github.com/rhysd/actionlint). CI runs the same script.

`.pre-commit-config.yaml` checks staged changes for secrets and formatting on
every commit; enable it with `prek install` (or `pre-commit install`). Never
commit tokens, passwords, keys or `.env` files; `.gitignore` excludes the
usual names, and gitleaks catches the rest. A false positive goes into
`.gitleaksignore`, not around the hook.

## Layout

| Path | Contents |
| --- | --- |
| `lib/PerCUDA.pm` | `new`/`run`, engine selection, user documentation (POD) |
| `lib/PerCUDA/Parser.pm` | Tokenizer and parser of the kernel language |
| `lib/PerCUDA/Analyzer.pm` | Argument kinds, scopes, type inference, kernel ABI |
| `lib/PerCUDA/CodeGen/CUDA.pm`, `Perl.pm` | Code for NVRTC and for the perl engine |
| `lib/PerCUDA/Compiler.pm` | NVRTC compilation and the on-disk cache |
| `lib/PerCUDA/Engine.pm`, `Engine/GPU.pm`, `Engine/Perl.pm` | Running kernels; `PerCUDA::Runtime` helpers of the generated Perl |
| `lib/PerCUDA/Array.pm` | Arrays kept with an engine between runs (`$runner->array`) |
| `lib/PerCUDA/Math.pm` | Reproducible math functions for `mode => 'exact'`, rendered to CUDA and Perl |
| `lib/PerCUDA/Data.pm` | Shapes and conversions of nested arrays |
| `t/` | Tests; `t/lib/PerCUDATest.pm` detects the GPU |
| `xt/` | Author tests (Perl::Critic, examples) |
| `eg/`, `author/` | Examples and benchmarks |

## Generated files

`Build.PL`, `META.json` and `README.md` come from Minilla; do not edit them
by hand. Change `cpanfile`, `minil.toml` or the POD in `lib/PerCUDA.pm`, then
run `minil build`.

## Style

- English only: code, comments, documentation, commit messages, pull
  requests and issues.
- `use v5.40` with signatures; no experimental features.
- Every option and public method is documented in POD and has tests,
  including error cases.
- A change to the kernel language or its semantics updates both code
  generators and the engine comparison tests.
- Record user-visible changes in `Changes`.
