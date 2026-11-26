package PerCUDA;
use v5.40;
use Carp          ();
use Cwd           ();
use Symbol        ();
use Scalar::Util  qw(looks_like_number);
use PerCUDA::Data ();
use PerCUDA::Parser;
use PerCUDA::Analyzer qw(infer_dim infer_types);
use PerCUDA::CodeGen::CUDA;
use PerCUDA::CodeGen::Perl;
use PerCUDA::Engine::Perl;
use PerCUDA::Engine::GPU;

our $VERSION = "0.01";

# Errors raised inside the engines point at the caller of run, not at PerCUDA.
our @CARP_NOT = qw(PerCUDA::Engine PerCUDA::Kernel PerCUDA::Context);

# The directory perl started in: kernel subs are read from their source file
# by the path perl compiled them from, which may be relative to it.
our $START_DIR = Cwd::getcwd();

sub new ($class, %args) {

    # The options of a context (internal: from PerCUDA::Context->kernel), then
    # those of a :GPU sub's attribute, then those given here; argument types
    # and constants merge by name, and a context fixes engine and device.
    my $shared    = delete $args{_context} // {};
    my $attribute = {};
    if (ref $args{source} eq 'CODE' && $INC{'PerCUDA/Kernel.pm'}) {
        $attribute    = { %{ PerCUDA::Kernel->attribute_args($args{source}) // {} } };
        $args{source} = $attribute->{source} if $attribute->{source};
        delete @$attribute{qw(source engine device)} if %$shared;
    }
    for my $level ($shared, $attribute, \%args) {
        Carp::croak("PerCUDA: 'constant' must be a hash reference of name => number")
            if defined $level->{constant} && ref $level->{constant} ne 'HASH';
    }
    my @argument = map  { @{ $_->{argument} // [] } } $attribute, \%args;
    my @constant = grep { defined } map { $_->{constant} } $attribute, \%args;

    # A context's argument types and constants apply to the kernels that use
    # those names (barewords for constants), not to every kernel.
    my %shared          = %$shared;
    my @shared_argument = @{ delete $shared{argument} // [] };
    my %shared_constant = %{ delete $shared{constant} // {} };
    %args = (
        %shared, %$attribute, %args,
        @argument ? (argument => \@argument)                : (),
        @constant ? (constant => { map { %$_ } @constant }) : (),
    );
    my $source = $args{source}   // Carp::croak("PerCUDA: 'source' is required");
    my $pairs  = $args{argument} // (ref $source || !@shared_argument ? undef : [@shared_argument]);
    Carp::croak("PerCUDA: 'argument' is required (unless the source is a code reference)")
        if !defined $pairs && ref $source ne 'CODE';
    $pairs //= [];
    Carp::croak("PerCUDA: 'argument' must be an array reference of name => type pairs")
        unless ref $pairs eq 'ARRAY' && @$pairs % 2 == 0;
    my @arguments = map { [ $pairs->[ $_ * 2 ], $pairs->[ $_ * 2 + 1 ] ] } 0 .. @$pairs / 2 - 1;

    # source => \&sub or sub (...) { ... }: the body is read from the source
    # file, the signature gives the arguments and their order, and those
    # without a type in 'argument' get a default one.
    my $package;    # of a kernel sub, for its use constant NAME => ...
    if (ref $source eq 'CODE') {
        require PerCUDA::Kernel;
        (my $names, my $body, $package) = PerCUDA::Kernel->source_of($source);
        my %param = map { $_ => 1 } @$names;
        my %type  = (
            (
                map { $param{ $shared_argument[ $_ * 2 ] } ? @shared_argument[ $_ * 2, $_ * 2 + 1 ] : () }
                    0 .. @shared_argument / 2 - 1
            ),
            map { @$_ } @arguments
        );
        for my $pair (@arguments) {
            Carp::croak("PerCUDA: argument '$pair->[0]' is not a parameter of the kernel sub")
                unless $param{ $pair->[0] };
        }
        @arguments = map { [ $_, $type{$_} ] } @$names;
        $source    = $body;
    }
    Carp::croak("PerCUDA: 'source' must be a string or a code reference") if ref $source;

    my $precision = $args{precision} // 'single';
    Carp::croak("PerCUDA: precision must be 'single' or 'double'") unless $precision =~ /\A(?:single|double)\z/;
    my $mode = $args{mode} // 'compat';
    Carp::croak("PerCUDA: mode must be 'fast', 'compat' or 'exact'") unless $mode =~ /\A(?:fast|compat|exact)\z/;
    my $perl_jobs = $args{perl_jobs} // 1;
    Carp::croak("PerCUDA: perl_jobs must be a positive integer or 'auto'")
        unless $perl_jobs =~ /\A(?:[1-9][0-9]*|auto)\z/;
    $perl_jobs = _cpu_count() if $perl_jobs eq 'auto';

    my $ast       = PerCUDA::Parser->new(source => $source)->parse;
    my $constants = $args{constant} // {};
    Carp::croak("PerCUDA: 'constant' must be a hash reference of name => number") unless ref $constants eq 'HASH';
    for my $name (keys %$constants) {
        Carp::croak("PerCUDA: constant '$name' is also an argument") if grep { $_->[0] eq $name } @arguments;
    }

    # A context's constant whose name the kernel uses for an argument or a
    # variable of its own only replaces the bareword.
    my %own = map { $_->[0] => 1 } @arguments;
    PerCUDA::Analyzer::walk($ast, sub ($n) { $own{ $n->{name} } = 1 if $n->{t} eq 'my' || $n->{t} eq 'foreach' });
    my %shared_bare = map { $_ => $shared_constant{$_} } grep { $own{$_} } keys %shared_constant;
    $constants = { (map { $_ => $shared_constant{$_} } grep { !$own{$_} } keys %shared_constant), %$constants };
    my $barewords =
        { (defined $package ? %{ PerCUDA::Kernel->package_constants($package, $ast) } : ()), %shared_bare };
    PerCUDA::Analyzer::substitute_constants($ast, $constants, $barewords);

    # The grid: the shape of an array argument, or sizes given as integers
    # and scalar argument names.
    my $size = $args{size};
    Carp::croak("PerCUDA: size must be an argument name or an array reference of 1 to 3 sizes")
        if ref $size && (ref $size ne 'ARRAY' || !@$size || @$size > 3);

    # Default types (only for a code reference source).
    my @untyped = ref $args{source} ? map { $_->[0] } grep { !defined $_->[1] } @arguments : ();
    if (@untyped) {
        my %size_names = map { $_ => 1 } grep { !/\A[0-9]+\z/ } ref $size ? @$size : ($size // ());
        my $types      = infer_types($ast, \@untyped, $precision eq 'double' ? 'd' : 'f', \%size_names);
        $_->[1] //= $types->{ $_->[0] } for @arguments;
    }

    # size => 'n' with an integer scalar n means size => ['n'] (for an unused
    # argument, the value passed to run decides)
    if (defined $size && !ref $size) {
        my ($pair) = grep { $_->[0] eq $size } @arguments;
        $size = [$size] if $pair && (PerCUDA::Data::type_code($pair->[1]) // '') eq 'i' && _used_as_scalar($ast, $size);
    }
    my $dim = $args{dim} // (ref $size ? scalar @$size : infer_dim($ast, [ map { $_->[0] } @arguments ]));
    Carp::croak("PerCUDA: size gives " . @$size . " dimension(s) but dim is $dim") if ref $size && @$size != $dim;

    my $an  = PerCUDA::Analyzer->new(ast => $ast, dim => $dim, arguments => \@arguments, precision => $precision);
    my %arg = map { $_->{name} => $_ } @{ $an->args };

    # By default run returns every array the kernel writes.
    my $retval = $args{retval} // [ map { $_->{name} } grep { $_->{kind} eq 'array' && $_->{written} } @{ $an->args } ];
    $retval = [$retval] unless ref $retval;
    for my $name (@$retval) {
        Carp::croak("PerCUDA: retval '$name' is not an argument") unless $arg{$name};
        Carp::croak("PerCUDA: retval '$name' must be an array used in the kernel")
            unless $arg{$name}{kind} eq 'array';
    }

    if (ref $size) {
        for my $s (@$size) {
            next if defined $s && $s =~ /\A[0-9]+\z/;
            Carp::croak("PerCUDA: size '" . ($s // '') . "' must be an integer or a scalar argument")
                unless defined $s && $arg{$s} && $arg{$s}{kind} ne 'array';
            Carp::croak("PerCUDA: size '$s' must be an int argument") unless $arg{$s}{type} eq 'i';
        }
    }
    my $candidates;
    if (!ref $size && !defined $size) {

        # By default the grid is the shape of the first non-empty array passed
        # to run, among: arrays the kernel reads with at least dim dimensions,
        # other arrays with at least dim dimensions (any arrays if none has).
        my @arrays = grep { $_->{kind} eq 'array' } @{ $an->args };
        my @full   = grep { $_->{rank} >= $dim } @arrays;
        my %seen;
        my @from = @full ? @full : @arrays;    # arrays with fewer dimensions only when nothing else
        $candidates = [ grep { !$seen{$_}++ } map { $_->{name} } (grep { $_->{read} } @from), @from ];
        Carp::croak("PerCUDA: 'size' is required when the kernel uses no arrays") unless @$candidates;
        $size = $candidates->[0];
    }
    elsif (!ref $size) {
        Carp::croak("PerCUDA: size '$size' is not an argument") unless $arg{$size};
        Carp::croak("PerCUDA: size '$size' must be an array") if $arg{$size}{kind} eq 'scalar';
    }

    my $engine = $args{engine} // PerCUDA::default_engine();
    Carp::croak("PerCUDA: engine must be 'gpu' or 'perl'") if defined $engine && $engine !~ /\A(?:gpu|perl)\z/;

    my $check = $args{check} ? 1 : 0;
    my $self  = bless {
        analyzer   => $an,
        arguments  => \@arguments,
        mode       => $mode,
        check      => $check,
        size       => $size,
        candidates => $candidates,
        precision  => $precision,
    }, $class;
    my %common = (analyzer => $an, size => $size, candidates => $candidates, retval => $retval, check => $check);

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
        : PerCUDA::Engine::Perl->new(%common, mode => $mode, jobs => $perl_jobs);
    return $self;
}

sub engine ($self) { return $self->{engine}->name }

# The arguments in run order with their types: (name => type, ...).
sub arguments ($self) {
    return map { $_->{name} => PerCUDA::Data::type_name($_->{type}) } @{ $self->{analyzer}->args };
}

# The engine when none is given: $ENV{PERCUDA_ENGINE} (gpu or perl), like
# Taichi's TI_ARCH, or undef for automatic selection.
sub default_engine () {
    my $engine = $ENV{PERCUDA_ENGINE};
    return defined $engine && length $engine ? $engine : undef;
}

# Functions of the :GPU subs' world, called as PerCUDA::name(...) or
# imported (use PerCUDA qw(:kernel configure array runner)), like
# Getopt::Long::Configure. PerCUDA->name(...) works as well.
my %EXPORT = map { $_ => 1 } qw(configure array runner);

sub _function_args (@args) {
    shift @args if @args && defined $args[0] && !ref $args[0] && $args[0] eq __PACKAGE__;
    return @args;
}

# $runner->array: on the runner's engine. array(...) as a function: on the
# engine of the caller's configuration (configure), like its :GPU subs.
sub array (@args) {
    my $self = $args[0];
    if (Scalar::Util::blessed($self) && $self->isa(__PACKAGE__)) {
        shift @args;
        return _make_array($self->{engine}, $self->{precision}, @args);
    }
    require PerCUDA::Kernel;
    return PerCUDA::Kernel->package_context(scalar caller)->array(_function_args(@args));
}

# use PerCUDA ':kernel': subs with the :GPU attribute run as kernels.
# use PerCUDA ':kernel' => { options }: also configure(options), like
# use Getopt::Long qw(:config ...).
# use PerCUDA qw(configure array runner): import those functions.
sub import ($class, @tags) {
    my $package = caller;
    while (@tags) {
        my $tag = shift @tags;
        if ($tag eq ':kernel') {
            require PerCUDA::Kernel;
            PerCUDA::Kernel->import_into($package);
            PerCUDA::Kernel->configure($package, shift @tags) if ref $tags[0] eq 'HASH';
        }
        elsif ($EXPORT{$tag}) {
            *{ Symbol::qualify_to_ref($tag, $package) } = __PACKAGE__->can($tag);
        }
        else {
            Carp::croak("PerCUDA: unknown import '$tag' (use ':kernel', 'configure', 'array' or 'runner')");
        }
    }
    return;
}

# Options of the caller's :GPU subs and of array(), decided at run time:
# PerCUDA::configure(engine => $engine, precision => 'double').
sub configure (@options) {
    require PerCUDA::Kernel;
    return PerCUDA::Kernel->configure(scalar caller, _function_args(@options));
}

# The runner behind a :GPU sub, e.g. for ->array, ->arguments or ->cuda_source.
sub runner (@args) {
    my ($code) = _function_args(@args);
    require PerCUDA::Kernel;
    return PerCUDA::Kernel->runner_of($code);
}

sub context ($class, %defaults) {
    require PerCUDA::Context;
    return PerCUDA::Context->new(%defaults);
}

# PerCUDA->array and PerCUDA::Context->array on $engine. Without a type, an
# ndarray of an integer type is an int one, of float or double keeps its type,
# and anything else is a float (double with precision => 'double').
my %NDARRAY_TYPE = (
    float  => 'f',
    double => 'd',
    map { $_ => 'i' } qw(byte sbyte short ushort long ulong longlong ulonglong indx),
);

sub _make_array ($engine, $precision, @args) {
    my $data = @args % 2 ? shift @args : undef;
    my %args = @args;
    my $type;
    if (defined $args{type}) {
        $type = PerCUDA::Data::type_code($args{type})
            // Carp::croak("PerCUDA: array type must be 'int', 'float' or 'double', not '$args{type}'");
    }
    else {
        $type = PerCUDA::Data::is_ndarray($data) ? $NDARRAY_TYPE{ $data->type . '' } : undef;
        $type //= ($precision // 'single') eq 'double' ? 'd' : 'f';
    }
    my ($source, $dims, $rank) = $engine->array_shape($data, $args{dims});
    return $engine->make_array($source, $dims, $rank, $type);
}

sub _used_as_scalar ($ast, $name) {
    my $used = 0;
    PerCUDA::Analyzer::walk($ast, sub ($n) { $used = 1 if $n->{t} eq 'var' && $n->{name} eq $name });
    return $used;
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
    return $self->{cuda_source} //=
        PerCUDA::CodeGen::CUDA->new($self->{analyzer}, mode => $self->{mode}, check => $self->{check})->generate;
}

sub perl_source ($self) {
    return $self->{perl_source} //=
        PerCUDA::CodeGen::Perl->new($self->{analyzer}, mode => $self->{mode}, check => $self->{check})->generate;
}

sub run ($self, @values) {
    my $want = @{ $self->{arguments} };

    # Named when the first value is a string that is not a number, or is the
    # name of an argument (such as inf) and the count does not fit positions.
    my $first = $values[0];
    @values = $self->_named_values(@values)
        if defined $first
        && !ref $first
        && (!looks_like_number($first) || (@values != $want && grep { $_->[0] eq $first } @{ $self->{arguments} }));
    Carp::croak("PerCUDA: run expects $want arguments but got " . scalar @values) unless @values == $want;
    my @results = $self->{engine}->run(@values);
    return wantarray ? @results : $results[0];
}

# run(name => value, ...): the values in declaration order. An array the
# kernel never reads may be omitted (it starts as zeros in the shape of the
# grid), and so may an unused argument.
sub _named_values ($self, @pairs) {
    Carp::croak('PerCUDA: run with named arguments expects name => value pairs') if @pairs % 2;
    my %value;
    while (my ($name, $value) = splice @pairs, 0, 2) {
        Carp::croak("PerCUDA: run got an unknown argument '$name'")
            unless grep { $_->[0] eq $name } @{ $self->{arguments} };
        Carp::croak("PerCUDA: run got argument '$name' twice") if exists $value{$name};
        $value{$name} = $value;
    }

    # an explicit size must be given; an inferred one comes from any array
    my %needed =
          $self->{candidates} ? ()
        : ref $self->{size}   ? map { $_ => 1 } @{ $self->{size} }
        :                       ($self->{size} => 1);
    my @values;
    for my $arg (@{ $self->{analyzer}->args }) {
        my $name = $arg->{name};
        if (exists $value{$name}) {
            push @values, $value{$name};
        }
        elsif ($arg->{kind} eq 'unused' && !$needed{$name}) {
            push @values, undef;
        }
        elsif ($arg->{kind} eq 'array' && !$arg->{read} && !$needed{$name}) {
            push @values, [];
        }
        else {
            Carp::croak("PerCUDA: run is missing argument '$name'");
        }
    }
    return @values;
}

1;
__END__

=encoding utf-8

=for stopwords GPGPU NVRTC PerCUDA cubin Fukumoto Ishiura IEICE inf nan ms ndarray ndarrays ulp runner

=head1 NAME

PerCUDA - GPGPU framework for Perl: write CUDA kernels in Perl

=head1 SYNOPSIS

    use PerCUDA qw(:kernel configure array);

    # A kernel is a sub: its body runs once per element of the grid, on the GPU.
    sub matmul :GPU ($a, $b, $c, $n) {
        my $sum = 0.0;
        for (my $k = 0; $k < $n; $k++) {
            $sum += $a->[$x][$k] * $b->[$k][$y];
        }
        $c->[$x][$y] = $sum;
    }

    my $m = [ [ 1 .. 3 ], [ 4 .. 6 ], [ 7 .. 9 ] ];
    my $product = matmul($m, $m, [], 3);                # [ [ 30, 36, 42 ], ... ]
    my $same    = matmul(a => $m, b => $m, n => 3);     # named; outputs may be omitted

Nothing else needs saying: the types (float arrays, C<$n> an int as a loop
bound), the grid (2-dimensional because the kernel uses C<$y>, the shape of
C<$a>), the result (the written C<$c>) and the engine (the GPU when there is
one, otherwise perl, with the same results) all have defaults. Options go in
a hash after the types, if any:

    sub histogram :GPU(int, int, { retval => [] }) ($labels, $counts) { ... }

Options known only at run time apply to all the C<:GPU> subs of a package,
and device arrays keep data on the GPU between calls:

    configure(engine => $ARGV[0], precision => 'double');

    sub decay :GPU ($u, $rate) { $u->[$x][$y] *= 1.0 - $rate }

    my $u = array([ map { [ (1) x 512 ] } 1 .. 512 ]);    # double, on the device
    decay($u, 0.01) for 1 .. 100;
    my $total = $u->sum;

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

=item C<source> (required): the kernel, a sub (C<\&name> or
C<sub (...) { ... }>; see L</KERNEL SUBS>). A string also works (see
L</STRING KERNELS>).

=item C<argument>: C<< [ name => type, ... ] >>, the types of some
parameters (any order); the others get defaults (see L</Types>). Types are
C<int>, C<float> and C<double> (or C<i>, C<f> and C<d> for short). A
parameter used with subscripts in the kernel is an array (up to 3
dimensions); otherwise it is a scalar.

=item C<dim>: 1, 2 or 3. By default, the number of sizes in C<size> when
it is a list, otherwise 3 if the kernel uses C<$z> or C<$size_z>, 2 if it
uses C<$y> or C<$size_y>, and 1 otherwise (variables the kernel declares or
takes as arguments do not count).

=item C<size>: the grid size. Either the name of the array argument whose
shape it is, the name of an C<int> scalar argument (a 1-dimensional grid of
that size), or a list of sizes, each an integer or the name of an C<int>
scalar argument, e.g. C<< size => [ 'n', 3 ] >>. The scalar and list forms
let every array start empty, e.g. a kernel that only generates data.

By default, each C<run> takes the shape of the first array passed non-empty
among: the arrays the kernel reads that have at least C<dim> dimensions, the
other arrays with at least C<dim> dimensions (in declaration order within
each group; arrays with fewer dimensions only when no array has C<dim>).
When all of them are empty, C<run> dies instead of running no thread. So the output can come first and be passed empty,
as in C<< run([], $in) >>. Give C<size> when the grid is not the shape of
the first input, e.g. a kernel whose first input is a lookup table.

=item C<retval>: array arguments returned by C<run> (default: every array
the kernel writes, in declaration order; C<< retval => [] >> returns
nothing). On the GPU, each nested array or ndarray in C<retval> is copied
back after the run. When every C<retval> is a device array, C<run> returns
without waiting for the kernel, and a kernel error surfaces at the next
C<get> or reduction instead (with C<< retval => [] >>, C<run> waits).

=item C<constant>: C<< { NAME => number, ... } >>, constants of the kernel
in addition to those of the sub's package (C<use constant NAME => ...>),
which a kernel sub sees as barewords; C<constant> takes precedence. Each
C<NAME> (or C<$NAME>) in the kernel is replaced by the number: an integer in
32-bit range becomes an int literal, anything else a float literal (of the
C<precision>). A different value compiles a different kernel.

=item C<engine>: C<gpu> or C<perl>. When omitted, C<$ENV{PERCUDA_ENGINE}>
if set (as Taichi's C<TI_ARCH>), otherwise the GPU if L<CUDA::DriverAPI> and
a CUDA device are available.

=item C<precision>: C<single> (default) or C<double>, on both engines. The
type of float literals and of locals inferred as floating point (C<float> and
C<double> data keep their declared type). C<single> is faster on the GPU.

=item C<mode>: how closely the two engines agree (see
L</DIFFERENCES BETWEEN ENGINES>):

=over 4

=item C<fast>: each engine as fast as it can be. The perl engine skips single
precision rounding and 32-bit integer wrapping, so results differ slightly.

=item C<compat> (default): the GPU stays as fast as C<fast>; the perl engine
follows the GPU semantics, which makes it several times slower. Results are
identical unless the kernel calls C<exp>, C<log>, C<sin>, C<cos>, C<atan2> or
C<**> with a non-integer exponent: the GPU uses CUDA's functions and perl its
own, which can differ by about 1 ulp per call.

=item C<exact>: identical results, including those functions: both engines
compute them with PerCUDA's own implementation (L<PerCUDA::Math>), accurate to
a few ulp. Kernels using them run slower on the GPU (about 1.4 times in single
and 2.7 times in double precision for a kernel made of little else).

=back

=item C<perl_jobs>: number of processes for the perl engine, or C<auto> for
one per CPU (default 1). The rows of the grid (C<$x>) are dealt out to
forked processes, which send back their rows of the C<retval> arrays; results
are the same as with one process. Only kernels whose stores all use C<$x> as
the first index are split; others run in one process.

=item C<check>: true to check every array index (default off). An index out
of range on either engine makes C<run> die with the array, the dimension, the
index, the valid range, the kernel line and the thread, e.g.

    PerCUDA: index out of range at kernel line 3: $in, dimension 1, index 3 is not in 0 .. 2 (thread $x=0, $y=1)

Indices are truncated toward zero first, as in C. When several accesses are
out of range, the GPU reports one of them (threads run in parallel) and perl
the first in its order; the contents of the C<retval> arrays are then
unspecified. Checking slows kernels down (about 3 times for a stencil on the
GPU, 4 times on the perl engine) and costs nothing when off.

=item C<cache_dir>: directory of compiled kernels (C<$ENV{PERCUDA_CACHE_DIR}>,
default C<~/.cache/percuda>). A broken entry is compiled again on the first
run.

=item C<device>: CUDA device ordinal (default 0). Runners on the same device
share one CUDA context.

=back

=item C<< $runner->run(@values) >>

=item C<< $runner->run(name => $value, ...) >>

Runs the kernel with one value per declared argument, in order, or with
named values in any order. With names, an array the kernel never reads (and
that is not the C<size>) may be left out: it starts empty. Array
arguments are array references; an empty array reference takes the shape of
the grid and is filled with zeros. Returns the C<retval> arrays
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
C<retval> ndarray is reshaped and filled in place, fastest when it has the
type of its argument (C<long>, C<float> or C<double> for C<int>, C<float> or
C<double>); one of another type receives the values converted to its own
type. An empty one takes the shape of the grid. The perl engine
accepts ndarrays too, but unpacks them into nested arrays. PDL is never
loaded by PerCUDA itself.

=item C<< $runner->array($nested_or_ndarray, type => $type) >>

=item C<< $runner->array(type => $type, dims => \@dims) >>

Returns a L<PerCUDA::Array> that stays with the runner's engine between runs:
in device memory on the GPU, as a nested array already converted to C<$type>
on the perl engine. It is made from a nested array or ndarray (copied), or
filled with zeros in the shape C<@dims> (outermost first). Any runner on the
same engine accepts it in place of an array argument of the same type and
number of dimensions. C<type> is optional: an ndarray of an integer type
(C<byte>, C<long>, C<indx>, ...) makes an C<int> array, one of C<float> or
C<double> keeps its type, and anything else is a C<float> (a C<double> with
C<< precision => 'double' >>), like the default type of a kernel sub's arrays.

Runs use it in place: nothing is copied, a kernel that writes it changes it
(whether or not it is a C<retval>), and C<run> returns it for a C<retval>
without copying the result back. On the GPU, C<run> then returns before the
kernel finishes, and errors surface at the next C<< $array->get >>. Copy the
contents back with C<< $array->get >>. This removes the transfers from
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
them: C<< $array->sum >>, C<< $array->min >>, C<< $array->max >>,
C<< $array->argmin >> and C<< $array->argmax >> (the index of the first
minimum or maximum, outermost first, e.g. C<[ $i, $j ]>). The order of the
operations is fixed, so even floating point sums are identical on both
engines (except with C<< mode => 'fast' >>, where perl sums in its own
order). Integer sums are exact up to 64 bits; C<min> and C<max> ignore nan
unless every element is nan; reductions of an empty array give 0 (C<sum>)
or undef.

    my $hit = $runner->array(type => 'int', dims => [$samples]);
    $runner->run($hit, $seed);              # $hit->[$x] = random(...) < ... ? 1 : 0
    my $pi = 4 * $hit->sum / $samples;

=item C<< PerCUDA->context(%args) >>

Returns a L<PerCUDA::Context>: C<< $context->kernel(%more) >> is
C<< PerCUDA->new(%args, %more) >>, and C<< $context->array(...) >> makes device
arrays without a kernel. The engine is chosen once, so every kernel and
array of a context is on the same engine.

=item C<< $runner->arguments >>

The arguments in the order of C<run>, with their types:
C<< (name => type, ...) >>, e.g. to check the default types of a kernel sub.

=item C<< $runner->engine >>, C<< $runner->cuda_source >>, C<< $runner->perl_source >>

The engine in use, and the generated CUDA C++ and Perl code.

=back

=head1 FUNCTIONS

For C<:GPU> subs, called as C<PerCUDA::name(...)> or imported, like
L<Getopt::Long>'s C<Configure> (C<use PerCUDA qw(:kernel configure array runner)>):

=over 4

=item C<configure(%args)>

=item C<< use PerCUDA ':kernel' => { %args } >>

Options of C<new> (but the per-kernel C<source>, C<argument>, C<dim>,
C<size> and C<retval>) for every C<:GPU> sub of the calling package, decided
at run time, e.g. C<< configure(engine => $engine, precision => 'double') >>
(the global settings of Taichi's C<ti.init>, per package). The engine is
chosen once, here. The options in a sub's attribute take precedence (except
C<engine> and C<device>). A C<:GPU> sub builds its runner at its first call,
and again after the next C<configure>. The import form configures when the
module is compiled, as C<use Getopt::Long qw(:config ...)> does. Returns the
L<PerCUDA::Context> it makes (or takes, as C<configure($context)>).

=item C<array(...)>

A device array (as C<< $runner->array >>) on the engine and with the
precision of the calling package's configuration, so that its C<:GPU> subs
accept it.

=item C<runner(\&name)>

The runner of a C<:GPU> sub, e.g. for C<< ->arguments >>, C<< ->engine >> or
C<< ->cuda_source >>.

=back

=head1 KERNEL SUBS

A kernel can be written as a Perl sub with a signature of plain scalars,
whose body is in the kernel language. The signature gives the arguments and
the order of C<run>. Pass the sub to C<new> as C<< source => \&name >> or
C<< source => sub (...) { ... } >>:

    use PerCUDA ':kernel';

    sub saxpy ($a, $xs, $ys, $out) {
        $out->[$x] = $a * $xs->[$x] + $ys->[$x];
    }
    my $runner = PerCUDA->new(source => \&saxpy);
    my $result = $runner->run(2, [ 1, 2 ], [ 10, 20 ], []);    # [ 12, 24 ]

or give it the C<:GPU> attribute, and calling it runs the kernel (calling it
is C<< $runner->run(@_) >>):

    sub saxpy :GPU ($a, $xs, $ys, $out) {
        $out->[$x] = $a * $xs->[$x] + $ys->[$x];
    }
    my $result = saxpy(2, [ 1, 2 ], [ 10, 20 ], []);

=head2 Types

A kernel sub needs no types: each parameter without one gets a default.
Arrays are C<float> (C<double> with C<< precision => 'double' >>), and so are
scalars, except those used where CUDA needs an integer, which are C<int>: in
an array index, as an operand of C<%> or a bit operation, as an argument of
C<random()>, as the bound of a loop over an int, or as a C<size>. Single
precision is the default because consumer GPUs run C<double> many times
slower. C<< $runner->arguments >> lists the types in use. Give the others in
C<< argument => [ name => type, ... ] >> (any order, any subset), or all of
them, in order, in the C<:GPU> attribute:

    my $assign = PerCUDA->new(source => \&assign, argument => [ label => 'int', changed => 'int' ]);

    sub add :GPU(float, float, float) ($a, $b, $out) { ... }

=head2 Options

The C<:GPU> attribute takes the types, if any, then a hash of the options of
C<new>, written as for C<new> (C<argument>, C<dim>, C<size>, C<retval>,
C<constant>, C<precision>, C<mode>, C<engine>, C<check>, C<perl_jobs>,
C<cache_dir> and C<device>), like a function signature followed by keyword
arguments in Numba's C<@cuda.jit>:

    use constant SCALE => 0.5;

    sub scale :GPU({ precision => 'double' }) ($in, $out) {
        $out->[$x] = $in->[$x] * SCALE;
    }
    sub table :GPU(int, int, { size => [ 'n', 3 ] }) ($out, $n) { $out->[$x][$y] = $x * 10 + $y }

    my $scaled = scale([ 1, 2, 3 ], []);    # [ 0.5, 1, 1.5 ]
    my $t      = table([], 2);              # [ [ 0, 1, 2 ], [ 10, 11, 12 ] ]

The hash is evaluated when the sub is compiled, so its values are literals
(barewords are strings), not variables. Options known only at run time (an
engine from C<@ARGV>, say) go to C<configure>, which sets them
for all the C<:GPU> subs of the package; C<array> makes device
arrays to match:

    my ($engine, $n) = @ARGV;
    configure(engine => $engine, precision => 'double');

    sub step :GPU ($u, $v, $r) { ... }

    my ($u, $v) = map { array(dims => [ $n, $n ]) } 1 .. 2;
    for (1 .. 100) { step($u, $v, 0.24); ($u, $v) = ($v, $u) }

Anything not given (C<dim>, C<size>, C<retval>) is inferred as for C<new>.
Constants are best written with C<use constant>, as barewords: C<$NAME>
from C<< constant => { ... } >> would need C<our $NAME> to compile under
C<strict>.

=head2 Notes

In both forms, the sub is not sent to the GPU as a whole: its body is a
kernel, run once per element of the grid, in the kernel language (see
L</KERNEL LANGUAGE>). It is read from the source file (when it is passed to
C<new> or a C<:GPU> sub is compiled), so it must be in a file (not
C<perl -e> or a string C<eval>; see L</STRING KERNELS>); it never runs as
Perl, so it cannot see variables outside the sub (pass them as arguments, or
use constants). C<:kernel> makes C<$x>, C<$y>, C<$z> and C<$size_x>,
C<$size_y>, C<$size_z> package variables of the caller, so that the body
compiles under C<strict>. The runner of a C<:GPU> sub is built on the first
call; kernel line numbers in errors are lines of the file.
C<< runner(\&scale) >> returns it, e.g. for C<< ->array >>,
C<< ->arguments >> or C<< ->cuda_source >>, and
C<< PerCUDA->new(source => \&scale, ...) >> makes another one from the
attribute and the options given (which take precedence).

=head1 STRING KERNELS

Where a kernel sub cannot be read from a file (C<perl -e>, a string
C<eval>, a REPL) or the kernel is generated by a program, C<source> can be
the body as a string. C<argument> is then required and gives every
parameter with its type, in the order of C<run>, since nothing else names
them:

    my $runner = PerCUDA->new(
        argument => [ in => 'float', out => 'float', s => 'float' ],
        source   => '$out->[$x] = $in->[$x] * $s;',
    );

Everything else works as for a kernel sub; constants are only those of
C<constant>.

=head1 KERNEL LANGUAGE

=over 4

=item * Scalars declared with C<my>, scalar and array arguments
(C<< $a->[$i]->[$j] >> or C<< $a->[$i][$j] >>, up to 3 dimensions). Kernels
cannot declare arrays, hashes or strings.

=item * Operators: C<+ - * / % **>, C<++ -->,
C<& | ^ ~ << E<gt>E<gt>>, C<&& || ! and or not>, C<< == != < > <= >= >>,
C<?:> and assignment operators.

=item * Control: C<if>/C<elsif>/C<else>, C<unless>, C<for (;;)>,
C<for my $i (A .. B)>, C<while>, C<until>, C<last>, C<next>, C<return>
(ends the thread), and statement modifiers C<if>/C<unless>.

=item * Constants: a bareword such as C<PI> (or C<$PI>) is a constant from
C<constant>, or from C<use constant> in the package of a kernel sub.

=item * Functions: C<sqrt>, C<abs>, C<int>, C<exp>, C<log>, C<sin>, C<cos>,
C<atan2($y, $x)>, and POSIX's C<floor> and C<ceil>. As on the GPU, C<sqrt>
of a negative number is nan and C<log> of 0 is -inf instead of an error.
C<**> with an integer exponent is repeated multiplication (identical on both
engines in every mode); with another exponent it is C<pow>.

=item * C<random($seed, $k1, $k2, $k3)> (1 to 4 integer arguments): a uniform
random number in [0, 1), with 24 random bits in single and 53 in double
precision. It is a counter-based generator: the same arguments always give
the same number, so pass what makes each draw distinct, e.g.
C<random($seed, $x, $y, $step)>. It uses 32-bit integer arithmetic only (a
chain of Chris Wellons' triple32 hash), so both engines give the same numbers
in every mode. It is meant for simulations, not for cryptography.

=back

Local variables are typed by inference: a variable is an int unless a float
value is ever assigned to it. C</> always divides as floating point, as in
Perl.

=head1 DIFFERENCES BETWEEN ENGINES

The perl engine runs Perl code generated from the same kernel and follows the
GPU semantics where Perl would differ:

=over 4

=item * Single precision values (C<float> data and, by default, float locals) are
rounded to single precision after every operation. The GPU code is compiled
without fused multiply-add, so floating point results match exactly in both
precisions.

=item * Integers wrap around at 32 bits.

=item * Division by zero gives inf, -inf or nan, and modulus by zero gives 0.

=item * Comparisons and C<!> yield 0 or 1.

=item * Bit operations work on 32-bit signed integers.

=item * Values are converted like C casts: C<int> data is truncated and C<float>
data is rounded to single precision, including the arrays passed to C<run>.

=item * Empty arrays are filled with zeros, and only C<retval> arrays (by
default, every array the kernel writes) are changed for the caller (other
arrays are copied first when needed).

=item * C<exp>, C<log>, C<sin>, C<cos>, C<atan2> and C<pow>: with
C<< mode => 'exact' >>, both engines use L<PerCUDA::Math>, which is built from
the operations above only, so results are bit for bit identical. Otherwise the
GPU uses CUDA's functions (C<expf> relies on a hardware approximation that
cannot be reproduced elsewhere), and perl its own, about 1 ulp apart.

=back

With C<< mode => 'fast' >>, the first two are skipped.

Remaining differences:

=over 4

=item * Without C<< check => 1 >>, out of range or negative array indices are
not checked: the GPU reads garbage and may overwrite other data, while perl
reads 0 (or counts negative indices from the end) and grows the array on
writes. Use C<check> while developing a kernel.

=item * Perl keeps integral values as integers, so it loses the sign of a
negative zero (C<0.0 * -1>), and dividing by it gives inf instead of -inf.

=back

=head1 PERFORMANCE

On a GeForce RTX 4070 SUPER with a Core i7-14700F, a 1024 x 1024 single
precision matrix multiplication (C<eg/benchmark.pl>) takes per C<run> on the
GPU about 20 ms with nested arrays (13 ms when the output array is reused),
most of it converting the Perl arrays, and 2.2 ms with ndarrays, close to
hand-written CUDA. The perl engine takes 17 s with C<< perl_jobs => 'auto' >>
(11 s with C<< mode => 'fast' >>).

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

C<eg/> has samples of computations common in research, each checked against
a known answer (C<eg/README.md> lists their methods, parameters and
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
