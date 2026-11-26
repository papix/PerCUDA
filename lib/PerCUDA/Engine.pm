package PerCUDA::Engine;
use v5.40;
use List::Util   ();
use Carp         ();
use Scalar::Util qw(blessed refaddr);
use PerCUDA::Array;
use PerCUDA::Data qw(shape is_ndarray type_name);

our @CARP_NOT = qw(PerCUDA PerCUDA::Runtime);

# Base class of the engines. Both engines share the calling convention of
# run: one value per argument, empty arrays take the shape of the size
# argument and are zero filled, and only the returned arrays change for the caller
# (by default, every array the kernel writes; see PerCUDA->new).
# PerCUDA::Array values stay with the engine and are used in place.

sub new ($class, %args) {
    my $self = bless {%args}, $class;    # analyzer, size, returns (+ engine specific)
    my $an   = $self->{analyzer};
    $self->{args}        = $an->args;
    $self->{dim}         = $an->dim;
    $self->{index}       = { map { $self->{args}[$_]{name} => $_ } 0 .. $#{ $self->{args} } };
    $self->{is_returned} = { map { $_                      => 1 } @{ $self->{returns} } };
    $self->init;
    return $self;
}

sub init ($self) { $self->init_storage; return }

# An engine without a kernel, which only makes and reduces device arrays
# (PerCUDA::Context->array).
sub storage ($class, %args) {
    my $self = bless {%args}, $class;
    $self->init_storage;
    return $self;
}

sub init_storage ($self) { }

# The error of check => 1, the same on both engines.
# $thread: [ name => index, ... ] per axis, the names the kernel gives the
# thread index (undef where it declares none).
sub out_of_range ($name, $line, $thread, $axis, $index, $size) {
    my @pairs = List::Util::pairs(@$thread);
    my $where = join ', ', map { defined $_->[0] ? "\$$_->[0]=$_->[1]" : $_->[1] } @pairs;
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

# Whether an array argument has elements ([[]] has none).
sub _non_empty ($value, $rank) {
    return 1             if is_device_array($value);
    return $value->nelem if is_ndarray($value);
    for (1 .. $rank) {
        return 0 unless ref $value eq 'ARRAY' && @$value;
        $value = $value->[0];
    }
    return 1;
}

# Validates @values and returns
# { grid => [ size per axis ], arrays => { index => { dims => [...], empty, ndarray, device } } }
# Array arguments are nested array references, PDL ndarrays or
# PerCUDA::Array objects; dims are outermost first in all cases.
sub layout ($self, $values) {
    my $size_dims;
    if (ref $self->{size}) {    # integers and scalar argument names
        $size_dims = [
            map {
                my $n = /\A[0-9]+\z/ ? $_ : $values->[ $self->{index}{$_} ];
                Carp::croak("PerCUDA: size '$_' must be a non-negative integer, not " . ($n // 'undef'))
                    unless defined $n && !ref $n && $n =~ /\A\s*\+?[0-9]+\s*\z/;
                $n + 0;
            } @{ $self->{size} }
        ];
    }
    else {
        my $name = $self->{size};
        if ($self->{candidates}) {    # inferred: the first candidate passed non-empty
            ($name) =
                grep { _non_empty($values->[ $self->{index}{$_} ], $self->{args}[ $self->{index}{$_} ]{rank}) }
                @{ $self->{candidates} };
            Carp::croak('PerCUDA: no array with '
                    . $self->{dim}
                    . ' or more dimensions was passed with elements to give the grid ('
                    . join(', ', map { "'$_'" } @{ $self->{candidates} })
                    . " were empty); pass one, or give 'size'")
                unless defined $name;
        }
        my $size_index = $self->{index}{$name};
        my $size_arg   = $self->{args}[$size_index];
        my $size_value = $values->[$size_index];
        if (defined $size_value && !ref $size_value && $size_arg->{kind} eq 'unused') {    # a number: a 1-D grid
            Carp::croak("PerCUDA: size '$name' must be a non-negative integer, not $size_value")
                unless $size_value =~ /\A\s*\+?[0-9]+\s*\z/;
            Carp::croak("PerCUDA: size '$name' gives 1 dimension but "
                    . $self->{analyzer}->index_function
                    . " gives $self->{dim}")
                unless $self->{dim} == 1;
            $size_dims = [ $size_value + 0 ];
        }
        $size_dims //=
            is_device_array($size_value) ? $size_value->dims : shape($size_value, $size_arg->{rank} || $self->{dim});
        Carp::croak("PerCUDA: size argument '$name' has fewer than $self->{dim} dimensions")
            if @$size_dims < $self->{dim};
    }

    my (%arrays, %device_args);
    for my $i (0 .. $#{ $self->{args} }) {
        my $arg   = $self->{args}[$i];
        my $value = $values->[$i];
        if ($arg->{kind} eq 'array' && ref $value ne 'ARRAY' && is_device_array($value)) {
            Carp::croak("PerCUDA: array '$arg->{name}' belongs to the $value->{engine} engine, not " . $self->name)
                unless $value->{engine} eq $self->name;
            Carp::croak("PerCUDA: array '$arg->{name}' must be of type "
                    . type_name($arg->{type})
                    . ", not "
                    . type_name($value->{type}))
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
