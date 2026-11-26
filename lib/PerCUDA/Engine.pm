package PerCUDA::Engine;
use v5.40;
use Carp         ();
use Scalar::Util qw(blessed refaddr);
use PerCUDA::Array;
use PerCUDA::Data qw(shape is_ndarray check_ndarray_type);

our @CARP_NOT = qw(PerCUDA PerCUDA::Runtime);

# Base class of the engines. Both engines share the calling convention of
# run: one value per argument, empty arrays take the shape of the size
# argument and are zero filled, and only retval arrays change for the caller.
# PerCUDA::Array values stay with the engine and are used in place.

sub new ($class, %args) {
    my $self = bless {%args}, $class;    # analyzer, size, retval (+ engine specific)
    my $an   = $self->{analyzer};
    $self->{args}      = $an->args;
    $self->{dim}       = $an->dim;
    $self->{index}     = { map { $self->{args}[$_]{name} => $_ } 0 .. $#{ $self->{args} } };
    $self->{is_retval} = { map { $_                      => 1 } @{ $self->{retval} } };
    $self->init;
    return $self;
}

sub init ($self) { }

# The error of check => 1, the same on both engines.
sub out_of_range ($name, $line, $thread, $axis, $index, $size) {
    my @axes  = qw(x y z);
    my $where = join ', ', map { "\$$axes[$_]=$thread->[$_]" } 0 .. $#$thread;
    Carp::croak("PerCUDA: index out of range at kernel line $line: \$$name, dimension $axis, index $index"
            . ' is not in 0 .. '
            . ($size - 1)
            . " (thread $where)");
}

sub is_device_array ($value) { return blessed $value && $value->isa('PerCUDA::Array') }

# Arguments of PerCUDA->array: a nested array or ndarray, or dims for zeros.
# Returns (data or undef, dims, rank).
sub array_shape ($self, $data, $dims) {
    if (defined $data) {
        my $rank = 0;
        if (is_ndarray($data)) {
            $rank = $data->ndims;
        }
        else {
            for (my $node = $data; ref $node eq 'ARRAY' && $rank < 3; $node = $node->[0]) { $rank++ }
        }
        my $dims = $rank >= 1 && $rank <= 3 ? shape($data, $rank) : [];
        Carp::croak('PerCUDA: array needs a non-empty array reference or ndarray of 1 to 3 dimensions')
            unless @$dims && !grep { !$_ } @$dims;
        return ($data, $dims, $rank);
    }
    Carp::croak('PerCUDA: array needs data or dims => [ 1 to 3 sizes ]')
        unless ref $dims eq 'ARRAY' && @$dims >= 1 && @$dims <= 3 && !grep { !/\A[0-9]+\z/ } @$dims;
    return (undef, [@$dims], scalar @$dims);
}

# Validates @values and returns
# { grid => [ size per axis ], arrays => { index => { dims => [...], empty, ndarray, device } } }
# Array arguments are nested array references, PDL ndarrays or
# PerCUDA::Array objects; dims are outermost first in all cases.
sub layout ($self, $values) {
    my $size_index = $self->{index}{ $self->{size} };
    my $size_arg   = $self->{args}[$size_index];
    my $size_value = $values->[$size_index];
    my $size_dims =
        is_device_array($size_value) ? $size_value->dims : shape($size_value, $size_arg->{rank} || $self->{dim});
    Carp::croak("PerCUDA: size argument '$self->{size}' has fewer than $self->{dim} dimensions")
        if @$size_dims < $self->{dim};

    my (%arrays, %device_args);
    for my $i (0 .. $#{ $self->{args} }) {
        my $arg   = $self->{args}[$i];
        my $value = $values->[$i];
        if ($arg->{kind} eq 'array' && ref $value ne 'ARRAY' && is_device_array($value)) {
            Carp::croak("PerCUDA: array '$arg->{name}' belongs to the $value->{engine} engine, not " . $self->name)
                unless $value->{engine} eq $self->name;
            Carp::croak("PerCUDA: array '$arg->{name}' must be of type $arg->{type}, not $value->{type}")
                unless $value->{type} eq $arg->{type};
            Carp::croak("PerCUDA: array '$arg->{name}' must have $arg->{rank} dimensions")
                unless @{ $value->{dims} } == $arg->{rank};
            push @{ $device_args{ refaddr $value } }, $arg;
            $arrays{$i} = { dims => $value->dims, empty => 0, device => 1 };
        }
        elsif ($arg->{kind} eq 'array') {
            my $ndarray = ref $value ne 'ARRAY' && is_ndarray($value);
            Carp::croak("PerCUDA: argument '$arg->{name}' must be an array reference or a PDL ndarray")
                unless $ndarray || ref $value eq 'ARRAY';
            my $empty = $ndarray ? $value->nelem == 0 : !@$value;
            my $dims;
            if (!$empty) {
                Carp::croak("PerCUDA: ndarray '$arg->{name}' must have $arg->{rank} dimensions")
                    if $ndarray && $value->ndims != $arg->{rank};
                $dims = shape($value, $arg->{rank});
            }
            else {
                Carp::croak("PerCUDA: cannot infer the shape of empty array '$arg->{name}'")
                    if $arg->{rank} > @$size_dims;
                $dims = [ @{$size_dims}[ 0 .. $arg->{rank} - 1 ] ];
            }
            check_ndarray_type($value, $arg->{type}, $arg->{name}) if $ndarray && $self->{is_retval}{ $arg->{name} };
            $arrays{$i} = { dims => $dims, empty => $empty, ndarray => $ndarray };
        }
        elsif ($arg->{kind} eq 'scalar') {
            Carp::croak("PerCUDA: argument '$arg->{name}' must be a number") if ref $value;
        }
    }

    # Device arrays are used in place, so one that is written cannot also be
    # another argument: the threads would race on it.
    for my $args (values %device_args) {
        next unless @$args > 1 && grep { $_->{written} } @$args;
        Carp::croak('PerCUDA: the same array is passed for '
                . join(' and ', map { "'$_->{name}'" } @$args)
                . ', and the kernel writes it');
    }
    return { grid => [ @{$size_dims}[ 0 .. $self->{dim} - 1 ] ], arrays => \%arrays };
}

1;
