package PerCUDA::Engine::Perl;
use v5.40;
use Carp ();
use PerCUDA::Data qw(shape flatten unflatten);

# Runs the kernel in the perl interpreter. The kernel is compiled into a Perl
# function that loops over every thread index (Fig. 3(b) of the paper), with
# the GPU's semantics where they differ from Perl (see PerCUDA::CodeGen).
# Like the GPU engine, empty arrays are filled with zeros and only retval
# arrays are visible to the caller after run.

my $SERIAL = 0;

sub new ($class, %args) {
    my $self = bless {%args}, $class;    # dim, size, retval, args, perl
    my $package = 'PerCUDA::Engine::Perl::Kernel' . $SERIAL++;
    $self->{perl_source} = "package $package;\nuse strict;\nuse warnings;\nno warnings qw(uninitialized numeric);\n$args{perl}";
    $self->{code} = eval $self->{perl_source};    ## no critic (ProhibitStringyEval)
    Carp::croak("PerCUDA: failed to compile the kernel for the perl engine: $@\n$self->{perl_source}")
        unless $self->{code};
    return $self;
}

sub name ($self) { 'perl' }

sub perl_source ($self) { $self->{perl_source} }

sub run ($self, @values) {
    my @args  = @{ $self->{args} };
    my %index = map { $args[$_]{name} => $_ } 0 .. $#args;
    my %is_retval = map { $_ => 1 } @{ $self->{retval} };

    my $size_arg  = $args[ $index{ $self->{size} } ];
    my $size_dims = shape($values[ $index{ $self->{size} } ], $size_arg->{rank} || $self->{dim});
    Carp::croak("PerCUDA: size argument '$self->{size}' has fewer than $self->{dim} dimensions")
        if @$size_dims < $self->{dim};

    my @call = @values;
    for my $i (0 .. $#args) {
        my $arg = $args[$i];
        next unless $arg->{kind} eq 'array';
        my $value = $values[$i];
        Carp::croak("PerCUDA: argument '$arg->{name}' must be an array reference") unless ref $value eq 'ARRAY';
        if (!$is_retval{ $arg->{name} } && ($arg->{written} || !@$value)) {
            # the GPU never copies non-retval arrays back, so work on a copy
            $value = @$value ? unflatten(flatten($value, $arg->{rank})) : [];
            $call[$i] = $value;
        }
        if (!@$value) {
            Carp::croak("PerCUDA: cannot infer the shape of empty array '$arg->{name}'")
                if $arg->{rank} > @$size_dims;
            my @dims = @{$size_dims}[ 0 .. $arg->{rank} - 1 ];
            my $n = 1;
            $n *= $_ for @dims;
            @$value = @{ unflatten([ (0) x $n ], \@dims) };
        }
    }
    $self->{code}->(@call, @{$size_dims}[ 0 .. $self->{dim} - 1 ]);
    return map { $values[ $index{$_} ] } @{ $self->{retval} };
}

package PerCUDA::Runtime;
use integer;

# Bit operations on 32-bit signed integers, as on the GPU.
sub _i32 ($v) { unpack 'l', pack 'L', $v & 0xFFFFFFFF }

sub band ($l, $r) { _i32(_i32($l) & _i32($r)) }
sub bor  ($l, $r) { _i32(_i32($l) | _i32($r)) }
sub bxor ($l, $r) { _i32(_i32($l) ^ _i32($r)) }
sub bnot ($v)     { _i32(~_i32($v)) }
sub shl  ($l, $r) { _i32(_i32($l) << _i32($r)) }
sub shr  ($l, $r) { _i32($l) >> _i32($r) }

1;
