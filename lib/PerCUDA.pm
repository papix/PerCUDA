package PerCUDA;
use v5.40;
use Carp ();
use PerCUDA::Parser;
use PerCUDA::CodeGen;
use PerCUDA::Compiler;
use PerCUDA::Engine::Perl;
use PerCUDA::Engine::GPU;

our $VERSION = "0.01";

sub new ($class, %args) {
    my $source = $args{source} // Carp::croak("PerCUDA: 'source' is required");
    my $dim    = $args{dim} // 1;
    my $pairs  = $args{argument} // Carp::croak("PerCUDA: 'argument' is required");
    Carp::croak("PerCUDA: 'argument' must be an array reference of name => type pairs")
        unless ref $pairs eq 'ARRAY' && @$pairs % 2 == 0;
    my @arguments = map { [ $pairs->[ $_ * 2 ], $pairs->[ $_ * 2 + 1 ] ] } 0 .. @$pairs / 2 - 1;

    my $ast = PerCUDA::Parser->new(source => $source)->parse;
    my $gen = PerCUDA::CodeGen->new(ast => $ast, dim => $dim, arguments => \@arguments)->generate;
    my %arg = map { $_->{name} => $_ } @{ $gen->{args} };

    my $retval = $args{retval} // [];
    $retval = [$retval] unless ref $retval;
    for my $name (@$retval) {
        Carp::croak("PerCUDA: retval '$name' is not an argument") unless $arg{$name};
        Carp::croak("PerCUDA: retval '$name' must be an array used in the kernel")
            unless $arg{$name}{kind} eq 'array';
    }

    my $size = $args{size} // (map { $_->{name} } grep { $_->{kind} eq 'array' } @{ $gen->{args} })[0]
        // Carp::croak("PerCUDA: 'size' is required when the kernel uses no arrays");
    Carp::croak("PerCUDA: size '$size' is not an argument") unless $arg{$size};
    Carp::croak("PerCUDA: size '$size' must be an array") if $arg{$size}{kind} eq 'scalar';

    my $self = bless {
        dim       => $dim,
        size      => $size,
        retval    => $retval,
        arguments => \@arguments,
        c_source  => $gen->{c},
    }, $class;

    my $engine = $args{engine};
    Carp::croak("PerCUDA: engine must be 'gpu' or 'perl'") if defined $engine && $engine !~ /\A(?:gpu|perl)\z/;
    my $compiler = PerCUDA::Compiler->new(map { $_ => $args{$_} } grep { defined $args{$_} } qw(clang arch cache_dir));

    my $context;
    if (!defined $engine || $engine eq 'gpu') {
        $context = $compiler->available ? PerCUDA::Engine::GPU->context_available($args{device} // 0) : undef;
        if (!$context && defined $engine) {
            Carp::croak($compiler->available
                ? "PerCUDA: no usable GPU (CUDA::DriverAPI could not create a context)"
                : "PerCUDA: clang ($compiler->{clang}) was not found");
        }
        $engine = $context ? 'gpu' : 'perl';
    }

    if ($engine eq 'gpu') {
        $self->{ptx}    = $compiler->compile($gen->{c});
        $self->{engine} = PerCUDA::Engine::GPU->new(
            dim     => $dim,
            size    => $size,
            retval  => $retval,
            args    => $gen->{args},
            ptx     => $self->{ptx},
            context => $context,
        );
    }
    else {
        $self->{engine} = PerCUDA::Engine::Perl->new(
            dim       => $dim,
            size      => $size,
            retval    => $retval,
            arguments => \@arguments,
            source    => $source,
        );
    }
    return $self;
}

sub engine   ($self) { $self->{engine}->name }
sub c_source ($self) { $self->{c_source} }
sub ptx_path ($self) { $self->{ptx} }

sub run ($self, @values) {
    my $want = @{ $self->{arguments} };
    Carp::croak("PerCUDA: run expects $want arguments but got " . scalar @values) unless @values == $want;
    my @results = $self->{engine}->run(@values);
    return wantarray ? @results : $results[0];
}

1;
__END__

=encoding utf-8

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
into C, compiled to PTX by clang's NVPTX backend, and launched through
L<CUDA::DriverAPI>. The same kernel can also run in the perl interpreter, so
scripts can be developed and tested on machines without a GPU.

The kernel body runs once per thread. C<$x>, C<$y> and C<$z> hold the thread
index and C<$size_x>, C<$size_y> and C<$size_z> the grid size (up to C<dim>
axes). The grid size is the shape of the C<size> argument.

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

=item C<engine>: C<gpu> or C<perl>. When omitted, the GPU is used if clang and
a CUDA device are available.

=item C<arch>, C<clang>, C<cache_dir>, C<device>: PTX target (default
C<sm_75>, or C<$ENV{PERCUDA_ARCH}>), clang command (C<$ENV{PERCUDA_CLANG}>),
PTX cache directory (C<$ENV{PERCUDA_CACHE_DIR}>, default
C<~/.cache/percuda>) and CUDA device ordinal.

=back

=item C<< $runner->run(@values) >>

Runs the kernel with one value per declared argument, in order. Array
arguments are array references; an empty array reference takes the shape of
the C<size> argument and is filled with zeros. Returns the C<retval> arrays
(the first one in scalar context). The caller's retval arrays are also updated
in place.

=item C<< $runner->engine >>, C<< $runner->c_source >>, C<< $runner->ptx_path >>

The engine in use, the generated C code and the path of the compiled PTX.

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

The GPU computes with 32-bit C<int> and C<float> (unless C<d> is used) while
perl uses 64-bit integers and doubles, so results may differ slightly. Values
stored into an C<i> array are truncated on the GPU only. Out of range or
negative array indices are not checked on the GPU. Only C<retval> arrays are
copied back from the GPU, whereas the perl engine modifies the arrays passed
to C<run> directly.

=head1 DEPENDENCIES

Perl 5.40 or later, clang with the NVPTX backend, and L<CUDA::DriverAPI>
(only for the GPU engine).

=head1 LICENSE

Copyright (C) papix.

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

=head1 AUTHOR

papix E<lt>mail@papix.netE<gt>

=cut
