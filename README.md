# NAME

PerCUDA - GPGPU framework for Perl: write CUDA kernels in Perl

# SYNOPSIS

    use PerCUDA qw(configure array);

    # A kernel is a sub: its body runs once per element of the grid, on the GPU.
    sub matmul :GPU ($a, $b, $c, $n) {
        my ($i, $j) = thread_index();    # this thread's element
        my $sum = 0.0;
        for (my $k = 0; $k < $n; $k++) {
            $sum += $a->[$i][$k] * $b->[$k][$j];
        }
        $c->[$i][$j] = $sum;
    }

    my $m = [ [ 1 .. 3 ], [ 4 .. 6 ], [ 7 .. 9 ] ];
    my $product = matmul($m, $m, [], 3);                # [ [ 30, 36, 42 ], ... ]
    my $same    = matmul(a => $m, b => $m, n => 3);     # named; outputs may be omitted

Nothing else needs saying: the types (float arrays, `$n` an int as a loop
bound), the grid (2-dimensional because `thread_index()` gives two values, the shape of
`$a`), the result (the written `$c`) and the engine (the GPU when there is
one, otherwise perl, with the same results) all have defaults. Options go in
a hash after the types, if any:

    sub histogram :GPU(int, int, { retval => [] }) ($labels, $counts) { ... }

Options known only at run time apply to all the `:GPU` subs of a package,
and device arrays keep data on the GPU between calls:

    configure(engine => $ARGV[0], precision => 'double');

    sub decay :GPU ($u, $rate) { my ($i, $j) = thread_index(); $u->[$i][$j] *= 1.0 - $rate }

    my $u = array([ map { [ (1) x 512 ] } 1 .. 512 ]);    # double, on the device
    decay($u, 0.01) for 1 .. 100;
    my $total = $u->sum;

# DESCRIPTION

PerCUDA lets you write GPU kernels in Perl. The kernel source is translated
into CUDA C++, compiled with NVRTC for the device's own architecture, and
launched through [CUDA::DriverAPI](https://metacpan.org/pod/CUDA%3A%3ADriverAPI). Compiled code is cached on disk. The
same kernel can also run in the perl interpreter, with the same results, so
scripts can be developed and tested on machines without a GPU.

    kernel (Perl) -> AST -> CUDA C++ -> NVRTC -> cubin (cached) -> CUDA::DriverAPI -> GPU
                         -> Perl (thread loops, GPU semantics) -> perl interpreter

The kernel body runs once per thread, like the body of nested loops over
the grid run in parallel. Each thread reads its index with
`my ($i, $j) = thread_index();` and may read the grid size with
`my ($ni, $nj) = grid_size();` (see ["Thread index"](#thread-index)). The grid has as many
dimensions (1 to 3) as `thread_index()` gives values, and its size is the
shape of the `size` argument.

This is a rebuild of the framework described in "PerCUDA: CUDA Binding
Framework for Perl" (Fukumoto and Ishiura, IEICE Technical Report, 2014) and
"Perl meets GPU" (Perl Advent Calendar 2013).

# METHODS

- `PerCUDA->new(%args)`
    - `source` (required): the kernel, a sub (`\&name` or
    `sub (...) { ... }`; see ["KERNEL SUBS"](#kernel-subs)). A string also works (see
    ["STRING KERNELS"](#string-kernels)).
    - `argument`: `[ name => type, ... ]`, the types of some
    parameters (any order); the others get defaults (see ["Types"](#types)). Types are
    `int`, `float` and `double` (or `i`, `f` and `d` for short). A
    parameter used with subscripts in the kernel is an array (up to 3
    dimensions); otherwise it is a scalar.
    - `size`: the grid size. Either the name of the array argument whose
    shape it is, the name of an `int` scalar argument (a 1-dimensional grid of
    that size), or a list of sizes, each an integer or the name of an `int`
    scalar argument, e.g. `size => [ 'n', 3 ]`. The scalar and list forms
    let every array start empty, e.g. a kernel that only generates data. A
    list must have as many sizes as `thread_index()` gives values (a kernel
    without `thread_index()` or `grid_size()` takes its dimensions from the
    list, otherwise it is 1-dimensional).

        By default, each `run` takes the shape of the first array passed non-empty
        among: the arrays the kernel reads that have at least as many dimensions as
        the grid, the other arrays with that many (in declaration order within
        each group; arrays with fewer dimensions only when no array has enough).
        When all of them are empty, `run` dies instead of running no thread. So the output can come first and be passed empty,
        as in `run([], $in)`. Give `size` when the grid is not the shape of
        the first input, e.g. a kernel whose first input is a lookup table.

    - `retval`: array arguments returned by `run` (default: every array
    the kernel writes, in declaration order; `retval => []` returns
    nothing). On the GPU, each nested array or ndarray in `retval` is copied
    back after the run. When every `retval` is a device array, `run` returns
    without waiting for the kernel, and a kernel error surfaces at the next
    `get` or reduction instead (with `retval => []`, `run` waits).
    - `constant`: `{ NAME => number, ... }`, constants of the kernel
    in addition to those of the sub's package (`use constant NAME =` ...>),
    which a kernel sub sees as barewords; `constant` takes precedence. Each
    `NAME` (or `$NAME`) in the kernel is replaced by the number: an integer in
    32-bit range becomes an int literal, anything else a float literal (of the
    `precision`). A different value compiles a different kernel.
    - `engine`: `gpu` or `perl`. When omitted, `$ENV{PERCUDA_ENGINE}`
    if set (as Taichi's `TI_ARCH`), otherwise the GPU if [CUDA::DriverAPI](https://metacpan.org/pod/CUDA%3A%3ADriverAPI) and
    a CUDA device are available.
    - `precision`: `single` (default) or `double`, on both engines. The
    type of float literals and of locals inferred as floating point (`float` and
    `double` data keep their declared type). `single` is faster on the GPU.
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
    one per CPU (default 1). The rows of the grid (the first value of
    `thread_index()`) are dealt out to forked processes, which send back their
    rows of the `retval` arrays; results are the same as with one process. Only
    kernels whose stores all use that value, unchanged, as the first index are
    split; others run in one process.
    - `check`: true to check every array index (default off). An index out
    of range on either engine makes `run` die with the array, the dimension, the
    index, the valid range, the kernel line and the thread (by the names
    `thread_index()` gives it), e.g.

            PerCUDA: index out of range at kernel line 3: $in, dimension 1, index 3 is not in 0 .. 2 (thread $i=0, $j=1)

        Indices are truncated toward zero first, as in C. When several accesses are
        out of range, the GPU reports one of them (threads run in parallel) and perl
        the first in its order; the contents of the `retval` arrays are then
        unspecified. Checking slows kernels down (about 3 times for a stencil on the
        GPU, 4 times on the perl engine) and costs nothing when off.

    - `cache_dir`: directory of compiled kernels (`$ENV{PERCUDA_CACHE_DIR}`,
    default `~/.cache/percuda`). A broken entry is compiled again on the first
    run.
    - `device`: CUDA device ordinal (default 0). Runners on the same device
    share one CUDA context.
- `$runner->run(@values)`
- `$runner->run(name => $value, ...)`

    Runs the kernel with one value per declared argument, in order, or with
    named values in any order. With names, an array the kernel never reads (and
    that is not the `size`) may be left out: it starts empty. Array
    arguments are array references; an empty array reference takes the shape of
    the grid and is filled with zeros. Returns the `retval` arrays
    (the first one in scalar context). The caller's `retval` arrays are also updated
    in place.

    When calling `run` repeatedly, passing the same output array every time is
    faster than a new `[]`: the GPU engine fills it in place, reusing its rows
    and scalars.

    Any array argument can also be a [PDL](https://metacpan.org/pod/PDL) ndarray, mixed freely with nested
    arrays. Dimensions are listed outermost first in PerCUDA, which is the
    reverse of `$ndarray->dims`: `$a->[$i][$j]` in a kernel is
    `$ndarray->at($j, $i)`. The GPU engine copies ndarrays without any
    per-element conversion, which makes it many times faster than nested arrays
    (see ["PERFORMANCE"](#performance)); an input ndarray of another type is converted. A
    `retval` ndarray is reshaped and filled in place, fastest when it has the
    type of its argument (`long`, `float` or `double` for `int`, `float` or
    `double`); one of another type receives the values converted to its own
    type. An empty one takes the shape of the grid. The perl engine
    accepts ndarrays too, but unpacks them into nested arrays. PDL is never
    loaded by PerCUDA itself.

- `$runner->array($nested_or_ndarray, type => $type)`
- `$runner->array(type => $type, dims => \@dims)`

    Returns a [PerCUDA::Array](https://metacpan.org/pod/PerCUDA%3A%3AArray) that stays with the runner's engine between runs:
    in device memory on the GPU, as a nested array already converted to `$type`
    on the perl engine. It is made from a nested array or ndarray (copied), or
    filled with zeros in the shape `@dims` (outermost first). Any runner on the
    same engine accepts it in place of an array argument of the same type and
    number of dimensions. `type` is optional: an ndarray of an integer type
    (`byte`, `long`, `indx`, ...) makes an `int` array, one of `float` or
    `double` keeps its type, and anything else is a `float` (a `double` with
    `precision => 'double'`), like the default type of a kernel sub's arrays.

    Runs use it in place: nothing is copied, a kernel that writes it changes it
    (whether or not it is a `retval`), and `run` returns it for a `retval`
    without copying the result back. On the GPU, `run` then returns before the
    kernel finishes, and errors surface at the next `$array->get`. Copy the
    contents back with `$array->get`. This removes the transfers from
    repeated runs on the same data:

        my $cur  = $life->array($board, type => 'int');
        my $next = $life->array(type => 'int', dims => [ 1024, 1024 ]);
        for (1 .. 1000) {
            $life->run($cur, $next);
            ($cur, $next) = ($next, $cur);
        }
        my $final = $cur->get;

    The same array cannot be passed for two arguments when the kernel writes
    either of them.

    Device arrays also reduce all their elements, on the engine that holds
    them: `$array->sum`, `$array->min`, `$array->max`,
    `$array->argmin` and `$array->argmax` (the index of the first
    minimum or maximum, outermost first, e.g. `[ $i, $j ]`). The order of the
    operations is fixed, so even floating point sums are identical on both
    engines (except with `mode => 'fast'`, where perl sums in its own
    order). Integer sums are exact up to 64 bits; `min` and `max` ignore nan
    unless every element is nan; reductions of an empty array give 0 (`sum`)
    or undef.

        my $hit = $runner->array(type => 'int', dims => [$samples]);
        $runner->run($hit, $seed);              # $hit->[$i] = random(...) < ... ? 1 : 0
        my $pi = 4 * $hit->sum / $samples;

- `PerCUDA->context(%args)`

    Returns a [PerCUDA::Context](https://metacpan.org/pod/PerCUDA%3A%3AContext): `$context->kernel(%more)` is
    `PerCUDA->new(%args, %more)`, and `$context->array(...)` makes device
    arrays without a kernel. The engine is chosen once, so every kernel and
    array of a context is on the same engine.

- `$runner->arguments`

    The arguments in the order of `run`, with their types:
    `(name => type, ...)`, e.g. to check the default types of a kernel sub.

- `$runner->engine`, `$runner->cuda_source`, `$runner->perl_source`

    The engine in use, and the generated CUDA C++ and Perl code.

# FUNCTIONS

For `:GPU` subs, called as `PerCUDA::name(...)` or imported, like
[Getopt::Long](https://metacpan.org/pod/Getopt%3A%3ALong)'s `Configure` (`use PerCUDA qw(configure array runner)`):

- `configure(%args)`
- `use PerCUDA { %args }`

    Options of `new` (but the per-kernel `source`, `argument`,
    `size` and `retval`) for every `:GPU` sub of the calling package, decided
    at run time, e.g. `configure(engine => $engine, precision => 'double')`
    (the global settings of Taichi's `ti.init`, per package). The engine is
    chosen once, here. The options in a sub's attribute take precedence (except
    `engine` and `device`). A `:GPU` sub builds its runner at its first call,
    and again after the next `configure`. The import form configures when the
    module is compiled, as `use Getopt::Long qw(:config ...)` does. Returns the
    [PerCUDA::Context](https://metacpan.org/pod/PerCUDA%3A%3AContext) it makes (or takes, as `configure($context)`).

- `array(...)`

    A device array (as `$runner->array`) on the engine and with the
    precision of the calling package's configuration, so that its `:GPU` subs
    accept it.

- `runner(\&name)`

    The runner of a `:GPU` sub, e.g. for `->arguments`, `->engine` or
    `->cuda_source`.

# KERNEL SUBS

A kernel can be written as a Perl sub with a signature of plain scalars,
whose body is in the kernel language. The signature gives the arguments and
the order of `run`. Pass the sub to `new` as `source => \&name` or
`source => sub (...) { ... }`:

    use PerCUDA;

    sub saxpy ($a, $xs, $ys, $out) {
        my $i = thread_index();
        $out->[$i] = $a * $xs->[$i] + $ys->[$i];
    }
    my $runner = PerCUDA->new(source => \&saxpy);
    my $result = $runner->run(2, [ 1, 2 ], [ 10, 20 ], []);    # [ 12, 24 ]

or give it the `:GPU` attribute, and calling it runs the kernel (calling it
is `$runner->run(@_)`):

    sub saxpy :GPU ($a, $xs, $ys, $out) {
        my $i = thread_index();
        $out->[$i] = $a * $xs->[$i] + $ys->[$i];
    }
    my $result = saxpy(2, [ 1, 2 ], [ 10, 20 ], []);

## Types

A kernel sub needs no types: each parameter without one gets a default.
Arrays are `float` (`double` with `precision => 'double'`), and so are
scalars, except those used where CUDA needs an integer, which are `int`: in
an array index, as an operand of `%` or a bit operation, as an argument of
`random()`, as the bound of a loop over an int, or as a `size`. Single
precision is the default because consumer GPUs run `double` many times
slower. `$runner->arguments` lists the types in use. Give the others in
`argument => [ name => type, ... ]` (any order, any subset), or all of
them, in order, in the `:GPU` attribute:

    my $assign = PerCUDA->new(source => \&assign, argument => [ label => 'int', changed => 'int' ]);

    sub add :GPU(float, float, float) ($a, $b, $out) { ... }

## Options

The `:GPU` attribute takes the types, if any, then a hash of the options of
`new`, written as for `new` (`argument`, `size`, `retval`,
`constant`, `precision`, `mode`, `engine`, `check`, `perl_jobs`,
`cache_dir` and `device`), like a function signature followed by keyword
arguments in Numba's `@cuda.jit`:

    use constant SCALE => 0.5;

    sub scale :GPU({ precision => 'double' }) ($in, $out) {
        my $i = thread_index();
        $out->[$i] = $in->[$i] * SCALE;
    }
    sub table :GPU(int, int, { size => [ 'n', 3 ] }) ($out, $n) {
        my ($i, $j) = thread_index();
        $out->[$i][$j] = $i * 10 + $j;
    }

    my $scaled = scale([ 1, 2, 3 ], []);    # [ 0.5, 1, 1.5 ]
    my $t      = table([], 2);              # [ [ 0, 1, 2 ], [ 10, 11, 12 ] ]

The hash is evaluated when the sub is compiled, so its values are literals
(barewords are strings), not variables. Options known only at run time (an
engine from `@ARGV`, say) go to `configure`, which sets them
for all the `:GPU` subs of the package; `array` makes device
arrays to match:

    my ($engine, $n) = @ARGV;
    configure(engine => $engine, precision => 'double');

    sub step :GPU ($u, $v, $r) { ... }

    my ($u, $v) = map { array(dims => [ $n, $n ]) } 1 .. 2;
    for (1 .. 100) { step($u, $v, 0.24); ($u, $v) = ($v, $u) }

Anything not given (`size`, `retval`) is inferred as for `new`.
Constants are best written with `use constant`, as barewords: `$NAME`
from `constant => { ... }` would need `our $NAME` to compile under
`strict`.

## Notes

In both forms, the sub is not sent to the GPU as a whole: its body is a
kernel, run once per element of the grid, in the kernel language (see
["KERNEL LANGUAGE"](#kernel-language)). It is read from the source file (when it is passed to
`new` or a `:GPU` sub is compiled), so it must be in a file (not
`perl -e` or a string `eval`; see ["STRING KERNELS"](#string-kernels)); it never runs as
Perl, so it cannot see variables outside the sub (pass them as arguments, or
use constants). `use PerCUDA` enables the `:GPU` attribute in the calling
package and imports `thread_index` and `grid_size` into it, so that the
body compiles under `strict` (called outside a kernel, they die);
`use PerCUDA ()` does neither. The runner of a `:GPU` sub is built on the first
call; kernel line numbers in errors are lines of the file.
`runner(\&scale)` returns it, e.g. for `->array`,
`->arguments` or `->cuda_source`, and
`PerCUDA->new(source => \&scale, ...)` makes another one from the
attribute and the options given (which take precedence).

# STRING KERNELS

Where a kernel sub cannot be read from a file (`perl -e`, a string
`eval`, a REPL) or the kernel is generated by a program, `source` can be
the body as a string. `argument` is then required and gives every
parameter with its type, in the order of `run`, since nothing else names
them:

    my $runner = PerCUDA->new(
        argument => [ in => 'float', out => 'float', s => 'float' ],
        source   => 'my $i = thread_index(); $out->[$i] = $in->[$i] * $s;',
    );

Everything else works as for a kernel sub; constants are only those of
`constant`.

# KERNEL LANGUAGE

## Thread index

    my ($i, $j) = thread_index();    # this thread's index, one value per axis
    my ($ni, $nj) = grid_size();     # the grid size
    my $i = thread_index();          # 1-dimensional

No variable is declared for the kernel: it reads its index with `thread_index()` and
the grid size with `grid_size()`, as Numba's `cuda.grid(2)` and
`cuda.gridsize(2)` or Warp's `wp.tid()` do. The number of variables
(1 to 3) is the number of dimensions of the grid; when both appear, they
must give the same number. Each may appear once, as a statement of its own
at the top level of the kernel (not inside a block), and the variables exist
from there on. Any names will do; the last one runs along CUDA's x axis
(consecutive threads, coalesced access).

## Statements and expressions

- Scalars declared with `my`, scalar and array arguments
(`$a->[$i]->[$j]` or `$a->[$i][$j]`, up to 3 dimensions). Kernels
cannot declare arrays, hashes or strings.
- Operators: `+ - * / % **`, `++ --`,
`& | ^ ~ << >>`, `&& || ! and or not`, `== != < > <= >=`,
`?:` and assignment operators.
- Control: `if`/`elsif`/`else`, `unless`, `for (;;)`,
`for my $i (A .. B)`, `while`, `until`, `last`, `next`, `return`
(ends the thread), and statement modifiers `if`/`unless`.
- Constants: a bareword such as `PI` (or `$PI`) is a constant from
`constant`, or from `use constant` in the package of a kernel sub.
- Functions: `sqrt`, `abs`, `int`, `exp`, `log`, `sin`, `cos`,
`atan2($y, $x)`, and POSIX's `floor` and `ceil`. As on the GPU, `sqrt`
of a negative number is nan and `log` of 0 is -inf instead of an error.
`**` with an integer exponent is repeated multiplication (identical on both
engines in every mode); with another exponent it is `pow`.
- `random($seed, $k1, $k2, $k3)` (1 to 4 integer arguments): a uniform
random number in \[0, 1), with 24 random bits in single and 53 in double
precision. It is a counter-based generator: the same arguments always give
the same number, so pass what makes each draw distinct, e.g.
`random($seed, $i, $j, $step)`. It uses 32-bit integer arithmetic only (a
chain of Chris Wellons' triple32 hash), so both engines give the same numbers
in every mode. It is meant for simulations, not for cryptography.

Local variables are typed by inference: a variable is an int unless a float
value is ever assigned to it. `/` always divides as floating point, as in
Perl.

# DIFFERENCES BETWEEN ENGINES

The perl engine runs Perl code generated from the same kernel and follows the
GPU semantics where Perl would differ:

- Single precision values (`float` data and, by default, float locals) are
rounded to single precision after every operation. The GPU code is compiled
without fused multiply-add, so floating point results match exactly in both
precisions.
- Integers wrap around at 32 bits.
- Division by zero gives inf, -inf or nan, and modulus by zero gives 0.
- Comparisons and `!` yield 0 or 1.
- Bit operations work on 32-bit signed integers.
- Values are converted like C casts: `int` data is truncated and `float`
data is rounded to single precision, including the arrays passed to `run`.
- Empty arrays are filled with zeros, and only `retval` arrays (by
default, every array the kernel writes) are changed for the caller (other
arrays are copied first when needed).
- `exp`, `log`, `sin`, `cos`, `atan2` and `pow`: with
`mode => 'exact'`, both engines use [PerCUDA::Math](https://metacpan.org/pod/PerCUDA%3A%3AMath), which is built from
the operations above only, so results are bit for bit identical. Otherwise the
GPU uses CUDA's functions (`expf` relies on a hardware approximation that
cannot be reproduced elsewhere), and perl its own, about 1 ulp apart.

With `mode => 'fast'`, the first two are skipped.

Remaining differences:

- Without `check => 1`, out of range or negative array indices are
not checked: the GPU reads garbage and may overwrite other data, while perl
reads 0 (or counts negative indices from the end) and grows the array on
writes. Use `check` while developing a kernel.
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
thread overwrites (`$out->[$i][$j] = ...` with `thread_index()`'s
variables, at the top level of the kernel, before any `return`) is neither uploaded nor zero filled, and the
same array passed for two read-only arguments is uploaded once. Compiled
kernels and the CUDA context are reused, so only the first `new` and `run`
of a process pay for the CUDA initialization (about 0.15 s).

# EXAMPLES

`eg/` has samples of computations common in research, each checked against
a known answer (`eg/README.md` lists their methods, parameters and
references):

    perl eg/heat.pl [gpu|perl] [n] [steps]              # heat equation, FTCS
    perl eg/poisson-cg.pl [gpu|perl] [n] [tolerance]    # Poisson, conjugate gradient
    perl eg/gray-scott.pl [gpu|perl] [n] [steps] [image.pgm]    # reaction-diffusion
    perl eg/nbody.pl [gpu|perl] [bodies] [steps] [dt]   # gravity, leapfrog
    perl eg/ising.pl [gpu|perl] [L] [sweeps] [T ...]    # Ising model, Metropolis
    perl eg/black-scholes.pl [gpu|perl] [options]       # option pricing, closed form
    perl eg/mc-option.pl [gpu|perl] [paths]             # option pricing, Monte Carlo
    perl eg/kmeans.pl [gpu|perl] [points] [K]           # k-means clustering

Others:

    perl eg/matmul.pl [gpu|perl]
    perl eg/mandelbrot.pl [gpu|perl] [width] [max_iterations] > mandelbrot.pgm
    perl eg/benchmark.pl [gpu|perl] [n]
    perl eg/pi.pl [gpu|perl] [samples] [seed]       # Monte Carlo: random() and sum
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
