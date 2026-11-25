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

=item C<< $array->type >>, C<< $array->dims >>, C<< $array->engine >>

Its type (C<i>, C<f> or C<d>), dimensions (outermost first) and engine
(C<gpu> or C<perl>).

=back

=cut
