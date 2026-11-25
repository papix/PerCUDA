# NAME

PerCUDA - GPGPU framework for Perl: write CUDA kernels in Perl

# SYNOPSIS

```perl
use PerCUDA;

my $array = [
    [ 1 .. 3 ],
    [ 4 .. 6 ],
    [ 7 .. 9 ],
];

my $runner = PerCUDA->new(
    dim      => 2,          # dimensions of the thread grid
    size     => 'array1',   # argument whose shape decides the grid size
    engine   => 'gpu',      # 'gpu', 'perl' or omitted (auto detect)
    argument => [           # arguments and their types
        array1 => 'f',
        array2 => 'f',
        array3 => 'f',
        N      => 'i',
    ],
    retval => ['array3'],   # arrays returned by run
    source => <<'KERNEL',   # kernel function written in Perl
my $tmp = 0;
for (my $i = 0; $i < $N; $i++) {
    $tmp = $tmp
        + $array1->[$i]->[$y]
        * $array2->[$x]->[$i];
}
$array3->[$x]->[$y] = $tmp;
KERNEL
);

my $retval = $runner->run($array, $array, [], 3);
```

# DESCRIPTION

PerCUDA lets you write GPU kernels in Perl. It is a rebuild of the framework
described in "PerCUDA: CUDA Binding Framework for Perl" (Fukumoto and Ishiura,
IEICE Technical Report, 2014) and "Perl meets GPU" (Perl Advent Calendar 2013).

```
kernel (Perl) --parse--> AST --> CUDA C++ --NVRTC--> cubin (cached) --CUDA::DriverAPI--> GPU
                          \----> Perl (thread loops, GPU semantics) --> perl interpreter
```

- The kernel body runs once per thread. `$x`, `$y`, `$z` are the thread
  index and `$size_x`, `$size_y`, `$size_z` the grid size, which is the shape
  of the `size` argument.
- Arguments used with subscripts are arrays (up to 3 dimensions, row-major);
  the others are scalars. Types are `i` (int), `f` (float) and `d` (double).
- `run` takes one value per argument. An empty array reference (`[]`) takes the
  shape of the `size` argument. It returns the `retval` arrays and also updates
  them in place.
- With `engine => 'perl'` (or automatically when no GPU is found) the
  kernel runs in the perl interpreter, so scripts can be developed without a GPU.
  By default (`perl_mode => 'compat'`) the perl engine follows the GPU
  semantics (single precision rounding, 32-bit integers, inf/nan on division by
  zero, 0/1 comparisons, C-like conversions, only `retval` arrays change), so
  both engines give exactly the same results. `perl_mode => 'fast'` skips the
  costly rounding and wrapping: several times faster, slightly different.
- Both engines compute in single precision by default, which is fast on the
  GPU. `precision => 'double'` switches both to double precision.
- Compiled kernels are cached in `~/.cache/percuda` (`PERCUDA_CACHE_DIR`).
- Calling `run` repeatedly with the same output array is faster than a new `[]`.

See `perldoc PerCUDA` for the supported kernel syntax and the differences
between the engines.

# EXAMPLES

```
perl examples/matmul.pl [gpu|perl]
perl examples/mandelbrot.pl [gpu|perl] [width] [max_iterations] > mandelbrot.pgm
```

# DEPENDENCIES

- Perl 5.40 or later
- For the GPU engine: [CUDA::DriverAPI](https://github.com/papix/CUDA-DriverAPI)
  (Perl 5.40+, CUDA 12+ with NVRTC).

# SETUP

CUDA 12+ support of CUDA::DriverAPI lives on its `fix-modern-cuda` branch
(`master` does not build with modern CUDA). PerCUDA needs its array transfer,
`fill`, `load` and `compute_capability` (commit 3c863cf or later). `.perl-version` pins Perl 5.44.0
for plenv. Either use its build output directly:

```
DRV=/path/to/CUDA-DriverAPI.fix-modern-cuda
PERL5LIB=$DRV/blib/lib:$DRV/blib/arch prove -l t/
```

or install it (`CUDA_HOME` must contain `include/cuda.h` when building):

```
cd /path/to/CUDA-DriverAPI.fix-modern-cuda && perl Build.PL && ./Build && ./Build install
```

Without CUDA::DriverAPI the tests that need a GPU are skipped and kernels run
on the perl engine.

# LICENSE

Copyright (C) papix.

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

# AUTHOR

papix <mail@papix.net>
