package PerCUDA::Array;
use v5.40;
use Carp          ();
use PerCUDA::Data qw(is_ndarray);

# An array that stays with an engine between runs: in device memory for the
# gpu engine, as a nested array already converted to its type for the perl
# engine. Created by PerCUDA->array; passed to run in place of an array.

my %TYPE = (i => 'i32', f => 'f32', d => 'f64');

sub _new ($class, %args) {
    return bless {%args}, $class;    # engine, type, dims, and buffer (gpu) or data (perl)
}

sub type   ($self) { return $self->{type} }
sub dims   ($self) { return [ @{ $self->{dims} } ] }
sub engine ($self) { return $self->{engine} }

# Copies the contents back: a new nested array, or into the given nested
# array or ndarray (which must have the matching type), like run does for
# retval arrays.
sub get ($self, %args) {
    my $into = $args{into};
    if ($self->{engine} eq 'gpu') {
        return $self->{buffer}->read_array(type => $TYPE{ $self->{type} }, dims => $self->{dims}, into => $into)
            if defined $into;
        return $self->{buffer}->read_array(type => $TYPE{ $self->{type} }, dims => $self->{dims});
    }
    my $copy = _copy($self->{data}, scalar @{ $self->{dims} });
    return $copy unless defined $into;
    if (is_ndarray($into)) {
        my $result = PDL->pdl($into->type, $copy);
        $into->reshape(reverse @{ $self->{dims} });
        $into .= $result;
        return $into;
    }
    @$into = @$copy;
    return $into;
}

# Reductions over every element, in the same order on both engines (see
# PerCUDA::Reduce). sum of an int array is exact (64-bit); min and max ignore
# nan unless every element is nan; argmin and argmax return the index of the
# first minimum or maximum, outermost first.
sub sum    ($self) { return ($self->_reduce('sum'))[0] }
sub min    ($self) { return ($self->_reduce('min'))[0] }
sub max    ($self) { return ($self->_reduce('max'))[0] }
sub argmin ($self) { return $self->_index(($self->_reduce('argmin'))[1]) }
sub argmax ($self) { return $self->_index(($self->_reduce('argmax'))[1]) }

sub _reduce ($self, $op) { return $self->{owner}->reduce($self, $op) }

sub _index ($self, $flat) {
    return undef if $flat < 0;    ## no critic (ProhibitExplicitReturnUndef)
    my @index;
    for my $dim (reverse @{ $self->{dims} }) {
        unshift @index, $flat % $dim;
        $flat = int($flat / $dim);
    }
    return \@index;
}

sub _copy ($ref, $rank) {
    return [@$ref] if $rank == 1;
    return [ map { _copy($_, $rank - 1) } @$ref ];
}

1;
__END__

=encoding utf-8

=head1 NAME

PerCUDA::Array - Array kept with an engine between runs

=head1 SYNOPSIS

    my $cur  = $runner->array($board, type => 'i');
    my $next = $runner->array(type => 'i', dims => [ 1024, 1024 ]);
    for (1 .. 1000) {
        $runner->run($cur, $next);
        ($cur, $next) = ($next, $cur);
    }
    my $final = $cur->get;

=head1 DESCRIPTION

Created by L<PerCUDA/array>. See there for how runs use it.

=head1 METHODS

=over 4

=item C<< $array->get >>, C<< $array->get(into => $nested_or_ndarray) >>

Copies the contents back to Perl, as a new nested array or into the given
nested array or ndarray (an ndarray must have the matching type).

=item C<< $array->sum >>, C<< $array->min >>, C<< $array->max >>, C<< $array->argmin >>, C<< $array->argmax >>

Reductions over every element, computed by the engine that holds the array
in a fixed order (see L<PerCUDA/array>).

=item C<< $array->type >>, C<< $array->dims >>, C<< $array->engine >>

Its type (C<i>, C<f> or C<d>), dimensions (outermost first) and engine
(C<gpu> or C<perl>).

=back

=cut
