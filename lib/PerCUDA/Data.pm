package PerCUDA::Data;
use v5.40;
use Carp ();
use Exporter 'import';

our @EXPORT_OK = qw(shape zeros copy_array);

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

sub zeros ($dims) {
    return [ (0) x $dims->[0] ] if @$dims == 1;
    my @inner = @{$dims}[ 1 .. $#$dims ];
    return [ map { zeros(\@inner) } 1 .. $dims->[0] ];
}

sub copy_array ($ref, $rank) {
    return [@$ref] if $rank == 1;
    return [ map { copy_array($_, $rank - 1) } @$ref ];
}

1;
