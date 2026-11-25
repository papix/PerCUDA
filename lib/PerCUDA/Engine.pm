package PerCUDA::Engine;
use v5.40;
use Carp          ();
use PerCUDA::Data qw(shape is_ndarray check_ndarray_type);

# Base class of the engines. Both engines share the calling convention of
# run: one value per argument, empty arrays take the shape of the size
# argument and are zero filled, and only retval arrays change for the caller.

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

# Validates @values and returns
# { grid => [ size per axis ], arrays => { index => { dims => [...], empty => bool, ndarray => bool } } }
# Array arguments are nested array references or PDL ndarrays; dims are
# outermost first in both cases.
sub layout ($self, $values) {
    my $size_index = $self->{index}{ $self->{size} };
    my $size_arg   = $self->{args}[$size_index];
    my $size_dims  = shape($values->[$size_index], $size_arg->{rank} || $self->{dim});
    Carp::croak("PerCUDA: size argument '$self->{size}' has fewer than $self->{dim} dimensions")
        if @$size_dims < $self->{dim};

    my %arrays;
    for my $i (0 .. $#{ $self->{args} }) {
        my $arg   = $self->{args}[$i];
        my $value = $values->[$i];
        if ($arg->{kind} eq 'array') {
            my $ndarray = is_ndarray($value);
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
    return { grid => [ @{$size_dims}[ 0 .. $self->{dim} - 1 ] ], arrays => \%arrays };
}

1;
