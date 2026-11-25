package PerCUDA;
use v5.40;
use Carp ();
use PerCUDA::Parser;
use PerCUDA::Analyzer;
use PerCUDA::CodeGen::CUDA;
use PerCUDA::CodeGen::Perl;
use PerCUDA::Engine::Perl;
use PerCUDA::Engine::GPU;

our $VERSION = "0.01";

sub new ($class, %args) {
    my $source = $args{source}   // Carp::croak("PerCUDA: 'source' is required");
    my $dim    = $args{dim}      // 1;
    my $pairs  = $args{argument} // Carp::croak("PerCUDA: 'argument' is required");
    Carp::croak("PerCUDA: 'argument' must be an array reference of name => type pairs")
        unless ref $pairs eq 'ARRAY' && @$pairs % 2 == 0;
    my @arguments = map { [ $pairs->[ $_ * 2 ], $pairs->[ $_ * 2 + 1 ] ] } 0 .. @$pairs / 2 - 1;

    my $precision = $args{precision} // 'single';
    Carp::croak("PerCUDA: precision must be 'single' or 'double'") unless $precision =~ /\A(?:single|double)\z/;
    my $perl_mode = $args{perl_mode} // 'compat';
    Carp::croak("PerCUDA: perl_mode must be 'compat' or 'fast'") unless $perl_mode =~ /\A(?:compat|fast)\z/;
    my $perl_jobs = $args{perl_jobs} // 1;
    Carp::croak("PerCUDA: perl_jobs must be a positive integer or 'auto'")
        unless $perl_jobs =~ /\A(?:[1-9][0-9]*|auto)\z/;
    $perl_jobs = _cpu_count() if $perl_jobs eq 'auto';

    my $ast = PerCUDA::Parser->new(source => $source)->parse;
    my $an  = PerCUDA::Analyzer->new(ast => $ast, dim => $dim, arguments => \@arguments, precision => $precision);
    my %arg = map { $_->{name} => $_ } @{ $an->args };

    my $retval = $args{retval} // [];
    $retval = [$retval] unless ref $retval;
    for my $name (@$retval) {
        Carp::croak("PerCUDA: retval '$name' is not an argument") unless $arg{$name};
        Carp::croak("PerCUDA: retval '$name' must be an array used in the kernel")
            unless $arg{$name}{kind} eq 'array';
    }

    my $size = $args{size} // (map { $_->{name} } grep { $_->{kind} eq 'array' } @{ $an->args })[0]
        // Carp::croak("PerCUDA: 'size' is required when the kernel uses no arrays");
    Carp::croak("PerCUDA: size '$size' is not an argument") unless $arg{$size};
    Carp::croak("PerCUDA: size '$size' must be an array") if $arg{$size}{kind} eq 'scalar';

    my $engine = $args{engine};
    Carp::croak("PerCUDA: engine must be 'gpu' or 'perl'") if defined $engine && $engine !~ /\A(?:gpu|perl)\z/;

    my $self   = bless { analyzer => $an, arguments => \@arguments, perl_mode => $perl_mode }, $class;
    my %common = (analyzer => $an, size => $size, retval => $retval);

    my $context;
    if (!defined $engine || $engine eq 'gpu') {
        $context = PerCUDA::Engine::GPU->context_available($args{device} // 0);
        Carp::croak("PerCUDA: no usable GPU (CUDA::DriverAPI could not create a context)")
            if !$context && defined $engine;
    }

    $self->{engine} =
        $context
        ? PerCUDA::Engine::GPU->new(
        %common,
        context   => $context,
        source    => $self->cuda_source,
        precision => $precision,
        cache_dir => $args{cache_dir},
        )
        : PerCUDA::Engine::Perl->new(%common, mode => $perl_mode, jobs => $perl_jobs);
    return $self;
}

sub engine ($self) { return $self->{engine}->name }

sub array ($self, @args) {
    my $data = @args % 2 ? shift @args : undef;
    my %args = @args;
    my $type = $args{type} // Carp::croak("PerCUDA: array needs type => 'i', 'f' or 'd'");
    Carp::croak("PerCUDA: array type must be 'i', 'f' or 'd'") unless $type =~ /\A[ifd]\z/;
    my ($source, $dims, $rank) = $self->{engine}->array_shape($data, $args{dims});
    return $self->{engine}->make_array($source, $dims, $rank, $type);
}

# Number of online CPUs (Linux /proc, BSD and macOS sysctl), or 1.
sub _cpu_count () {
    if (open my $fh, '<', '/proc/cpuinfo') {
        my $count = grep { /^processor\s*:/ } <$fh>;
        close $fh;
        return $count if $count;
    }
    my $sysctl = `sysctl -n hw.ncpu 2>/dev/null` // '';
    return $sysctl =~ /\A([1-9][0-9]*)/ ? $1 : 1;
}

sub cuda_source ($self) {
    return $self->{cuda_source} //= PerCUDA::CodeGen::CUDA->new($self->{analyzer})->generate;
}

sub perl_source ($self) {
    return $self->{perl_source} //=
        PerCUDA::CodeGen::Perl->new($self->{analyzer}, mode => $self->{perl_mode})->generate;
}

sub run ($self, @values) {
    my $want = @{ $self->{arguments} };
    Carp::croak("PerCUDA: run expects $want arguments but got " . scalar @values) unless @values == $want;
    my @results = $self->{engine}->run(@values);
    return wantarray ? @results : $results[0];
}

1;
__END__

=encoding utf-8

=for stopwords GPGPU NVRTC PerCUDA cubin Fukumoto Ishiura IEICE inf nan ms ndarray ndarrays

=head1 NAME

PerCUDA - GPGPU framework for Perl: write CUDA kernels in Perl

=head1 SYNOPSIS

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

=head1 DESCRIPTION

PerCUDA lets you write GPU kernels in Perl. The kernel source is translated
into CUDA C++, compiled with NVRTC for the device's own architecture, and
launched through L<CUDA::DriverAPI>. Compiled code is cached on disk. The
same kernel can also run in the perl interpreter, with the same results, so
scripts can be developed and tested on machines without a GPU.

    kernel (Perl) -> AST -> CUDA C++ -> NVRTC -> cubin (cached) -> CUDA::DriverAPI -> GPU
                         -> Perl (thread loops, GPU semantics) -> perl interpreter

The kernel body runs once per thread. C<$x>, C<$y> and C<$z> hold the thread
index and C<$size_x>, C<$size_y> and C<$size_z> the grid size (up to C<dim>
axes). The grid size is the shape of the C<size> argument.

This is a rebuild of the framework described in "PerCUDA: CUDA Binding
Framework for Perl" (Fukumoto and Ishiura, IEICE Technical Report, 2014) and
"Perl meets GPU" (Perl Advent Calendar 2013).

=head1 METHODS

=over 4

=item C<< PerCUDA->new(%args) >>

=over 4

=item C<source> (required): the kernel body.

=item C<argument> (required): C<< [ name => type, ... ] >>. Types are C<i> (int),
C<f> (float) and C<d> (double). An argument used with subscripts in the
kernel is an array (up to 3 dimensions); otherwise it is a scalar.

=item C<dim>: 1, 2 or 3 (default 1).

=item C<size>: the array argument whose shape is the grid size (default: the
first array argument).

=item C<retval>: array arguments returned by C<run>.

=item C<engine>: C<gpu> or C<perl>. When omitted, the GPU is used if
L<CUDA::DriverAPI> and a CUDA device are available.

=item C<precision>: C<single> (default) or C<double>, on both engines. The
type of float literals and of locals inferred as floating point (C<f> and
C<d> data keep their declared type). C<single> is faster on the GPU.

=item C<perl_mode>: C<compat> (default) or C<fast>, for the perl engine.
C<compat> gives exactly the GPU's results (see L</DIFFERENCES BETWEEN ENGINES>)
but rounding every single precision operation makes it several times slower.
C<fast> skips single precision rounding and 32-bit integer wrapping, so
results may differ slightly from the GPU.

=item C<perl_jobs>: number of processes for the perl engine, or C<auto> for
one per CPU (default 1). The rows of the grid (C<$x>) are dealt out to
forked processes, which send back their rows of the C<retval> arrays; results
are the same as with one process. Only kernels whose stores all use C<$x> as
the first index are split; others run in one process.

=item C<cache_dir>: directory of compiled kernels (C<$ENV{PERCUDA_CACHE_DIR}>,
default C<~/.cache/percuda>). A broken entry is compiled again on the first
run.

=item C<device>: CUDA device ordinal (default 0). Runners on the same device
share one CUDA context.

=back

=item C<< $runner->run(@values) >>

Runs the kernel with one value per declared argument, in order. Array
arguments are array references; an empty array reference takes the shape of
the C<size> argument and is filled with zeros. Returns the C<retval> arrays
(the first one in scalar context). The caller's C<retval> arrays are also updated
in place.

When calling C<run> repeatedly, passing the same output array every time is
faster than a new C<[]>: the GPU engine fills it in place, reusing its rows
and scalars.

Any array argument can also be a L<PDL> ndarray, mixed freely with nested
arrays. Dimensions are listed outermost first in PerCUDA, which is the
reverse of C<< $ndarray->dims >>: C<< $a->[$x][$y] >> in a kernel is
C<< $ndarray->at($y, $x) >>. The GPU engine copies ndarrays without any
per-element conversion, which makes it many times faster than nested arrays
(see L</PERFORMANCE>); an input ndarray of another type is converted. A
C<retval> ndarray must already have the type of its argument (C<long>,
C<float> or C<double> for C<i>, C<f> or C<d>) and is reshaped and filled in
place; an empty one takes the shape of the C<size> argument. The perl engine
accepts ndarrays too, but unpacks them into nested arrays. PDL is never
loaded by PerCUDA itself.

=item C<< $runner->array($nested_or_ndarray, type => $type) >>

=item C<< $runner->array(type => $type, dims => \@dims) >>

Returns a L<PerCUDA::Array> that stays with the runner's engine between runs:
in device memory on the GPU, as a nested array already converted to C<$type>
on the perl engine. It is made from a nested array or ndarray (copied), or
filled with zeros in the shape C<@dims> (outermost first). Any runner on the
same engine accepts it in place of an array argument of the same type and
number of dimensions.

Runs use it in place: nothing is copied, a kernel that writes it changes it
(whether or not it is a C<retval>), and C<run> returns it for a C<retval>
without copying the result back. On the GPU, C<run> then returns before the
kernel finishes, and errors surface at the next C<< $array->get >>. Copy the
contents back with C<< $array->get >>. This removes the transfers from
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

=item C<< $runner->engine >>, C<< $runner->cuda_source >>, C<< $runner->perl_source >>

The engine in use, and the generated CUDA C++ and Perl code.

=back

=head1 KERNEL LANGUAGE

=over 4

=item * Scalars declared with C<my>, scalar and array arguments
(C<< $a->[$i]->[$j] >> or C<< $a->[$i][$j] >>, up to 3 dimensions). Kernels
cannot declare arrays, hashes or strings.

=item * Operators: C<+ - * / % **> (integer exponent), C<++ -->,
C<& | ^ ~ << E<gt>E<gt>>, C<&& || ! and or not>, C<< == != < > <= >= >>,
C<?:> and assignment operators.

=item * Control: C<if>/C<elsif>/C<else>, C<unless>, C<for (;;)>,
C<for my $i (A .. B)>, C<while>, C<until>, C<last>, C<next>, C<return>
(ends the thread), and statement modifiers C<if>/C<unless>.

=item * Functions: C<sqrt>, C<abs>, C<int>.

=back

Local variables are typed by inference: a variable is an int unless a float
value is ever assigned to it. C</> always divides as floating point, as in
Perl.

=head1 DIFFERENCES BETWEEN ENGINES

The perl engine runs Perl code generated from the same kernel and follows the
GPU semantics where Perl would differ:

=over 4

=item * Single precision values (C<f> data and, by default, float locals) are
rounded to single precision after every operation. The GPU code is compiled
without fused multiply-add, so floating point results match exactly in both
precisions.

=item * Integers wrap around at 32 bits.

=item * Division by zero gives inf, -inf or nan, and modulus by zero gives 0.

=item * Comparisons and C<!> yield 0 or 1.

=item * Bit operations work on 32-bit signed integers.

=item * Values are converted like C casts: C<i> data is truncated and C<f>
data is rounded to single precision, including the arrays passed to C<run>.

=item * Empty arrays are filled with zeros, and only C<retval> arrays are
changed for the caller (other arrays are copied first when needed).

=back

With C<perl_mode =E<gt> 'fast'>, the first two are skipped.

Remaining differences:

=over 4

=item * Out of range or negative array indices are not checked on the GPU,
while perl reads 0 (or counts negative indices from the end) and grows the
array on writes.

=item * Perl keeps integral values as integers, so it loses the sign of a
negative zero (C<0.0 * -1>), and dividing by it gives inf instead of -inf.

=back

=head1 PERFORMANCE

On a GeForce RTX 4070 SUPER with a Core i7-14700F, a 1024 x 1024 single
precision matrix multiplication (C<eg/benchmark.pl>) takes per C<run> on the
GPU about 20 ms with nested arrays (13 ms when the output array is reused),
most of it converting the Perl arrays, and 2.2 ms with ndarrays, close to
hand-written CUDA. The perl engine takes 17 s with C<< perl_jobs => 'auto' >>
(11 s with C<< perl_mode => 'fast' >>).

Repeated runs on the same data gain the most from device arrays. For
Conway's Game of Life on a 1024 x 1024 board (C<author/bench-life.pl>), a
step takes 7.5 ms with nested arrays, 0.95 ms with ndarrays and 0.08 ms with
device arrays (C<< $runner->array >>); the perl engine takes 325 ms with
C<< perl_jobs => 'auto' >>.

The GPU engine skips work it can prove unnecessary: an output that every
thread overwrites (C<< $out->[$x][$y] = ... >> at the top level of the
kernel, before any C<return>) is neither uploaded nor zero filled, and the
same array passed for two read-only arguments is uploaded once. Compiled
kernels and the CUDA context are reused, so only the first C<new> and C<run>
of a process pay for the CUDA initialization (about 0.15 s).

=head1 EXAMPLES

    perl eg/matmul.pl [gpu|perl]
    perl eg/mandelbrot.pl [gpu|perl] [width] [max_iterations] > mandelbrot.pgm
    perl eg/benchmark.pl [gpu|perl] [n]
    perl author/bench-life.pl [--size N] [--steps N] [nested|ndarray|device|perl ...]

=head1 DEPENDENCIES

Perl 5.40 or later. The GPU engine needs L<CUDA::DriverAPI> with NVRTC (CUDA
12.0 or later) and an NVIDIA GPU; without them, kernels run on the perl
engine. L<PDL> is optional; passing ndarrays to the GPU engine needs a
CUDA::DriverAPI that accepts them in C<write_array> and C<read_array>.

=head1 SEE ALSO

L<CUDA::DriverAPI>, L<https://github.com/papix/CUDA-DriverAPI>,
L<https://qiita.com/papix/items/fee3e60e21f2ab1f1c76>,
L<https://ist.ksc.kwansei.ac.jp/~ishiura/publications/T2014-01b.pdf>

=head1 LICENSE

Copyright (C) papix.

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

=head1 AUTHOR

papix E<lt>mail@papix.netE<gt>

=cut
