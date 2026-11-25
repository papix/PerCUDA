package PerCUDA::Data;
use v5.40;
use Carp ();
use Exporter 'import';

our @EXPORT_OK = qw(shape flatten unflatten);

# Dimensions of a rectangular nested array, following the first element down to $rank levels.
sub shape ($ref, $rank) {
    my @dims;
    my $node = $ref;
    for my $level (1 .. $rank) {
        Carp::croak("PerCUDA: expected a $rank-dimensional array reference")
            unless ref $node eq 'ARRAY';
        push @dims, scalar @$node;
        $node = $node->[0];
    }
    return \@dims;
}

# Flattens a rectangular nested array in row-major order. Returns (\@flat, \@dims).
sub flatten ($ref, $rank) {
    my $dims = shape($ref, $rank);
    my @flat;
    if ($rank == 1) {
        @flat = @$ref;
    }
    elsif ($rank == 2) {
        for my $row (@$ref) {
            _check_row($row, $dims->[1]);
            push @flat, @$row;
        }
    }
    else {
        for my $plane (@$ref) {
            _check_row($plane, $dims->[1]);
            for my $row (@$plane) {
                _check_row($row, $dims->[2]);
                push @flat, @$row;
            }
        }
    }
    return (\@flat, $dims);
}

sub _check_row ($row, $len) {
    Carp::croak("PerCUDA: arrays must be rectangular (expected rows of $len elements)")
        unless ref $row eq 'ARRAY' && @$row == $len;
}

sub unflatten ($flat, $dims) {
    my @dims = @$dims;
    return [@$flat] if @dims == 1;
    my $inner = 1;
    $inner *= $_ for @dims[ 1 .. $#dims ];
    my @out;
    for my $i (0 .. $dims[0] - 1) {
        my @chunk = @{$flat}[ $i * $inner .. ($i + 1) * $inner - 1 ];
        push @out, unflatten(\@chunk, [ @dims[ 1 .. $#dims ] ]);
    }
    return \@out;
}

1;
