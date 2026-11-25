# AGENTS.md

Setup, commands, layout and style are in `CONTRIBUTING.md`; read it first.

## Checks

- `prove -l t/` after every change; `author/lint.sh` (run `author/lint.sh --fix`
  to apply perltidy) and `minil test --release` before committing.
- GPU tests skip silently without CUDA::DriverAPI or a device. A pass on a
  machine without a GPU, including CI, only proves the perl engine. Check
  `perl -MCUDA::DriverAPI -e1` first when GPU work matters.

## Decisions and pitfalls

- English only (the maintainer's decision): code, docs, commit messages, pull
  requests and issues. `author/lint.sh` flags Japanese text in files.
- No experimental Perl features (the maintainer's decision); stable `use v5.40`
  features only.
- The perl engine gives exactly the GPU's results in `perl_mode => 'compat'`
  (the maintainer's decision): single precision rounding after every
  operation, 32-bit integer wrap, inf/nan on division by zero, C-like
  conversions. Any change to `CodeGen::CUDA` needs the matching change in
  `CodeGen::Perl`; `t/04_semantics.t`, `t/05_mandelbrot.t` and
  `t/06_parity.t` compare the engines exactly.
- The GPU code is compiled with `--fmad=false`; fused multiply-add would break
  the exact match.
- The kernel ABI is defined once, in `PerCUDA::Analyzer::kernel_params`; the
  CUDA generator and the GPU engine both follow it.
- The last Perl axis is CUDA's x axis (coalesced access); launch dimensions
  in `Engine::GPU` are reversed accordingly.
- `Build.PL`, `META.json`, `README.md` are generated: edit `cpanfile`,
  `minil.toml` or the POD in `lib/PerCUDA.pm`, then `minil build`. Minilla
  does not copy dotfiles into its build directory, so author-test config
  lives in `xt/` (e.g. `xt/perlcriticrc`). perltidy skips the generated
  `Build.PL`; never reformat it.
- PDL is optional (the maintainer's decision): array arguments take nested
  arrays or ndarrays, nothing in `lib/` loads PDL (ndarrays are detected with
  `PerCUDA::Data::is_ndarray`), and `t/09_without_pdl.t` hides it. Dimensions
  are outermost first everywhere in PerCUDA and CUDA::DriverAPI, the reverse
  of `$ndarray->dims`; never transpose, and never convert nested arrays to
  ndarrays internally (slower than `write_array`).
- `PerCUDA::Analyzer` proves which work can be skipped: `covered` arrays
  (never read, stored unconditionally by every thread into its own element)
  are not uploaded, and `writes_own_rows` kernels may be split across
  `perl_jobs` processes. Keep those proofs conservative; `t/02_codegen.t` and
  `t/06_parity.t` hold the counterexamples.
- `PerCUDA::Array` (device arrays) is used in place on both engines: no
  copies, no conversion, writes visible even outside retval, and `run`
  returns without waiting on the GPU. Engine code must check `device` in the
  layout before any nested-array or ndarray handling.
- `perl_jobs` forks: children must leave with `POSIX::_exit` so that they run
  no END blocks or destructors (e.g. of the CUDA context) of the parent.
- All runners for a device share one CUDA context, so a sticky kernel error
  breaks every runner in the process.
- The driver loads code lazily: a broken cache entry loads fine and fails at
  the first `run`, where `Engine::GPU` recompiles and retries once.
- CUDA::DriverAPI (papix/CUDA-DriverAPI) is developed alongside; the GPU
  engine uses its `buffer`, `write_array`, `read_array`, `fill`, `compile`,
  `load`, `compute_capability` and `run` with `[type => value]` arguments.
  Check its POD before relying on anything else.
