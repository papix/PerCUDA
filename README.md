# NAME

PerCUDA - GPGPU framework for Perl: write CUDA kernels in Perl

# SYNOPSIS

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

# DESCRIPTION

PerCUDA lets you write GPU kernels in Perl. The kernel source is translated
into CUDA C++, compiled with NVRTC for the device's own architecture, and
launched through [CUDA::DriverAPI](https://metacpan.org/pod/CUDA%3A%3ADriverAPI). Compiled code is cached on disk. The
same kernel can also run in the perl interpreter, with the same results, so
scripts can be developed and tested on machines without a GPU.

    kernel (Perl) -> AST -> CUDA C++ -> NVRTC -> cubin (cached) -> CUDA::DriverAPI -> GPU
                         -> Perl (thread loops, GPU semantics) -> perl interpreter

The kernel body runs once per thread. `$x`, `$y` and `$z` hold the thread
index and `$size_x`, `$size_y` and `$size_z` the grid size (up to `dim`
axes). The grid size is the shape of the `size` argument.

This is a rebuild of the framework described in "PerCUDA: CUDA Binding
Framework for Perl" (Fukumoto and Ishiura, IEICE Technical Report, 2014) and
"Perl meets GPU" (Perl Advent Calendar 2013).

# METHODS

- `PerCUDA->new(%args)`
    - `source` (required): the kernel body.
    - `argument` (required): `[ name => type, ... ]`. Types are `i` (int),
    `f` (float) and `d` (double). An argument used with subscripts in the
    kernel is an array (up to 3 dimensions); otherwise it is a scalar.
    - `dim`: 1, 2 or 3 (default 1).
    - `size`: the array argument whose shape is the grid size (default: the
    first array argument).
    - `retval`: array arguments returned by `run`.
    - `engine`: `gpu` or `perl`. When omitted, the GPU is used if
    [CUDA::DriverAPI](https://metacpan.org/pod/CUDA%3A%3ADriverAPI) and a CUDA device are available.
    - `precision`: `single` (default) or `double`, on both engines. The
    type of float literals and of locals inferred as floating point (`f` and
    `d` data keep their declared type). `single` is faster on the GPU.
    - `mode`: how closely the two engines agree (see
    ["DIFFERENCES BETWEEN ENGINES"](#differences-between-engines)):
        - `fast`: each engine as fast as it can be. The perl engine skips single
        precision rounding and 32-bit integer wrapping, so results differ slightly.
        - `compat` (default): the GPU stays as fast as `fast`; the perl engine
        follows the GPU semantics, which makes it several times slower. Results are
        identical unless the kernel calls `exp`, `log`, `sin`, `cos`, `atan2` or
        `**` with a non-integer exponent: the GPU uses CUDA's functions and perl its
        own, which can differ by about 1 ulp per call.
        - `exact`: identical results, including those functions: both engines
        compute them with PerCUDA's own implementation ([PerCUDA::Math](https://metacpan.org/pod/PerCUDA%3A%3AMath)), accurate to
        a few ulp. Kernels using them run slower on the GPU (about 1.4 times in single
        and 2.7 times in double precision for a kernel made of little else).
    - `perl_jobs`: number of processes for the perl engine, or `auto` for
    one per CPU (default 1). The rows of the grid (`$x`) are dealt out to
    forked processes, which send back their rows of the `retval` arrays; results
    are the same as with one process. Only kernels whose stores all use `$x` as
    the first index are split; others run in one process.
    - `cache_dir`: directory of compiled kernels (`$ENV{PERCUDA_CACHE_DIR}`,
    default `~/.cache/percuda`). A broken entry is compiled again on the first
    run.
    - `device`: CUDA device ordinal (default 0). Runners on the same device
    share one CUDA context.
- `$runner->run(@values)`

    Runs the kernel with one value per declared argument, in order. Array
    arguments are array references; an empty array reference takes the shape of
    the `size` argument and is filled with zeros. Returns the `retval` arrays
    (the first one in scalar context). The caller's `retval` arrays are also updated
    in place.

    When calling `run` repeatedly, passing the same output array every time is
    faster than a new `[]`: the GPU engine fills it in place, reusing its rows
    and scalars.

    Any array argument can also be a [PDL](https://metacpan.org/pod/PDL) ndarray, mixed freely with nested
    arrays. Dimensions are listed outermost first in PerCUDA, which is the
    reverse of `$ndarray->dims`: `$a->[$x][$y]` in a kernel is
    `$ndarray->at($y, $x)`. The GPU engine copies ndarrays without any
    per-element conversion, which makes it many times faster than nested arrays
    (see ["PERFORMANCE"](#performance)); an input ndarray of another type is converted. A
    `retval` ndarray must already have the type of its argument (`long`,
    `float` or `double` for `i`, `f` or `d`) and is reshaped and filled in
    place; an empty one takes the shape of the `size` argument. The perl engine
    accepts ndarrays too, but unpacks them into nested arrays. PDL is never
    loaded by PerCUDA itself.

- `$runner->array($nested_or_ndarray, type => $type)`
- `$runner->array(type => $type, dims => \@dims)`

    Returns a [PerCUDA::Array](https://metacpan.org/pod/PerCUDA%3A%3AArray) that stays with the runner's engine between runs:
    in device memory on the GPU, as a nested array already converted to `$type`
    on the perl engine. It is made from a nested array or ndarray (copied), or
    filled with zeros in the shape `@dims` (outermost first). Any runner on the
    same engine accepts it in place of an array argument of the same type and
    number of dimensions.

    Runs use it in place: nothing is copied, a kernel that writes it changes it
    (whether or not it is a `retval`), and `run` returns it for a `retval`
    without copying the result back. On the GPU, `run` then returns before the
    kernel finishes, and errors surface at the next `$array->get`. Copy the
    contents back with `$array->get`. This removes the transfers from
    repeated runs on the same data:

        my $cur  = $life->array($board, type => 'i');
        my $next = $life->array(type => 'i', dims => [ 1024, 1024 ]);
        for (1 .. 1000) {
            $life->run($cur, $next);
            ($cur, $next) = ($next, $cur);
        }
        my $final = $cur->get;

    The same array cannot be passed for two arguments when the kernel writes
    either of them.

- `$runner->engine`, `$runner->cuda_source`, `$runner->perl_source`

    The engine in use, and the generated CUDA C++ and Perl code.

# KERNEL LANGUAGE

- Scalars declared with `my`, scalar and array arguments
(`$a->[$i]->[$j]` or `$a->[$i][$j]`, up to 3 dimensions). Kernels
cannot declare arrays, hashes or strings.
- Operators: `+ - * / % **`, `++ --`,
`& | ^ ~ << >>`, `&& || ! and or not`, `== != < > <= >=`,
`?:` and assignment operators.
- Control: `if`/`elsif`/`else`, `unless`, `for (;;)`,
`for my $i (A .. B)`, `while`, `until`, `last`, `next`, `return`
(ends the thread), and statement modifiers `if`/`unless`.
- Functions: `sqrt`, `abs`, `int`, `exp`, `log`, `sin`, `cos`,
`atan2($y, $x)`, and POSIX's `floor` and `ceil`. As on the GPU, `sqrt`
of a negative number is nan and `log` of 0 is -inf instead of an error.
`**` with an integer exponent is repeated multiplication (identical on both
engines in every mode); with another exponent it is `pow`.

Local variables are typed by inference: a variable is an int unless a float
value is ever assigned to it. `/` always divides as floating point, as in
Perl.

# DIFFERENCES BETWEEN ENGINES

The perl engine runs Perl code generated from the same kernel and follows the
GPU semantics where Perl would differ:

- Single precision values (`f` data and, by default, float locals) are
rounded to single precision after every operation. The GPU code is compiled
without fused multiply-add, so floating point results match exactly in both
precisions.
- Integers wrap around at 32 bits.
- Division by zero gives inf, -inf or nan, and modulus by zero gives 0.
- Comparisons and `!` yield 0 or 1.
- Bit operations work on 32-bit signed integers.
- Values are converted like C casts: `i` data is truncated and `f`
data is rounded to single precision, including the arrays passed to `run`.
- Empty arrays are filled with zeros, and only `retval` arrays are
changed for the caller (other arrays are copied first when needed).
- `exp`, `log`, `sin`, `cos`, `atan2` and `pow`: with
`mode => 'exact'`, both engines use [PerCUDA::Math](https://metacpan.org/pod/PerCUDA%3A%3AMath), which is built from
the operations above only, so results are bit for bit identical. Otherwise the
GPU uses CUDA's functions (`expf` relies on a hardware approximation that
cannot be reproduced elsewhere), and perl its own, about 1 ulp apart.

With `mode => 'fast'`, the first two are skipped.

Remaining differences:

- Out of range or negative array indices are not checked on the GPU,
while perl reads 0 (or counts negative indices from the end) and grows the
array on writes.
- Perl keeps integral values as integers, so it loses the sign of a
negative zero (`0.0 * -1`), and dividing by it gives inf instead of -inf.

# PERFORMANCE

On a GeForce RTX 4070 SUPER with a Core i7-14700F, a 1024 x 1024 single
precision matrix multiplication (`eg/benchmark.pl`) takes per `run` on the
GPU about 20 ms with nested arrays (13 ms when the output array is reused),
most of it converting the Perl arrays, and 2.2 ms with ndarrays, close to
hand-written CUDA. The perl engine takes 17 s with `perl_jobs => 'auto'`
(11 s with `mode => 'fast'`).

Repeated runs on the same data gain the most from device arrays. For
Conway's Game of Life on a 1024 x 1024 board (`author/bench-life.pl`), a
step takes 7.5 ms with nested arrays, 0.95 ms with ndarrays and 0.08 ms with
device arrays (`$runner->array`); the perl engine takes 325 ms with
`perl_jobs => 'auto'`.

The GPU engine skips work it can prove unnecessary: an output that every
thread overwrites (`$out->[$x][$y] = ...` at the top level of the
kernel, before any `return`) is neither uploaded nor zero filled, and the
same array passed for two read-only arguments is uploaded once. Compiled
kernels and the CUDA context are reused, so only the first `new` and `run`
of a process pay for the CUDA initialization (about 0.15 s).

# EXAMPLES

    perl eg/matmul.pl [gpu|perl]
    perl eg/mandelbrot.pl [gpu|perl] [width] [max_iterations] > mandelbrot.pgm
    perl eg/benchmark.pl [gpu|perl] [n]
    perl author/bench-life.pl [--size N] [--steps N] [nested|ndarray|device|perl ...]
    perl author/bench-math.pl [--size N] [--perl-size N]    # every mode

# DEPENDENCIES

Perl 5.40 or later. The GPU engine needs [CUDA::DriverAPI](https://metacpan.org/pod/CUDA%3A%3ADriverAPI) with NVRTC (CUDA
12.0 or later) and an NVIDIA GPU; without them, kernels run on the perl
engine. [PDL](https://metacpan.org/pod/PDL) is optional; passing ndarrays to the GPU engine needs a
CUDA::DriverAPI that accepts them in `write_array` and `read_array`.

# SEE ALSO

[CUDA::DriverAPI](https://metacpan.org/pod/CUDA%3A%3ADriverAPI), [https://github.com/papix/CUDA-DriverAPI](https://github.com/papix/CUDA-DriverAPI),
[https://qiita.com/papix/items/fee3e60e21f2ab1f1c76](https://qiita.com/papix/items/fee3e60e21f2ab1f1c76),
[https://ist.ksc.kwansei.ac.jp/~ishiura/publications/T2014-01b.pdf](https://ist.ksc.kwansei.ac.jp/~ishiura/publications/T2014-01b.pdf)

# LICENSE

Copyright (C) papix.

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

# AUTHOR

papix <mail@papix.net>
