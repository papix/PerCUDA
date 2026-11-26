package PerCUDA::Data;
use v5.40;
use Carp ();
use Exporter 'import';
use Scalar::Util qw(blessed);

our @EXPORT_OK =
    qw(shape zeros copy_array convert_array is_ndarray ndarray_has_type type_code type_name removed_option);

# Options that no longer exist, with what to do instead; every entry point
# (new, :GPU options, context, configure) reports them the same way.
my %REMOVED_OPTION = (
    dim    => "dim is not an option; the kernel's my (\$x, ...) = thread_index() gives the dimensions",
    retval => 'retval is now called returns',
);

# The error message for a removed option, or undef.
sub removed_option ($key) { return $REMOVED_OPTION{$key} }

# PDL type of each PerCUDA type. PDL itself is never loaded here: an ndarray
# can only arrive when the caller has loaded it.
my %PDL_TYPE = (i => 'long', f => 'float', d => 'double');

# Type names: int, float and double, or i, f and d for short. Internally
# PerCUDA uses the short codes.
my %TYPE_CODE = (int => 'i',   float => 'f',     double => 'd', i => 'i', f => 'f', d => 'd');
my %TYPE_NAME = (i   => 'int', f     => 'float', d => 'double');

# The code (i, f or d) of a type name, or undef if it is not one.
sub type_code ($name) { return defined $name && !ref $name ? $TYPE_CODE{$name} : undef }

# The name (int, float or double) of a type code.
sub type_name ($code) { return $TYPE_NAME{$code} // $code }

sub is_ndarray ($value) { return blessed $value && $value->isa('PDL') }

# Whether an ndarray holds the PerCUDA type $type (long, float or double).
sub ndarray_has_type ($ndarray, $type) { return $ndarray->type eq $PDL_TYPE{$type} }

# Dimensions of a rectangular nested array, following the first element down
# to $rank levels, outermost first. For a PDL ndarray this is the reverse of
# ->dims (PDL lists the innermost dimension first; the memory order is the same).
sub shape ($ref, $rank) {
    if (is_ndarray($ref)) {
        my @dims = reverse $ref->dims;
        Carp::croak("PerCUDA: ndarray must have at least $rank dimensions, not " . scalar(@dims))
            if @dims < $rank;
        return [ @dims[ 0 .. $rank - 1 ] ];
    }
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

# Copy of a nested array with every element converted the way the GPU stores
# it: truncated for i, rounded to single precision for f, as is for d.
sub convert_array ($ref, $rank, $type) {
    if ($rank == 1) {
        no warnings 'uninitialized';
        return [ unpack 'f*', pack 'f*', @$ref ] if $type eq 'f';
        return [ map { int } @$ref ]             if $type eq 'i';
        return [@$ref];
    }
    return [ map { convert_array($_, $rank - 1, $type) } @$ref ];
}

1;
