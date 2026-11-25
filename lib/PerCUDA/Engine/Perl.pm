package PerCUDA::Engine::Perl;
use v5.40;
use Carp ();
use PerCUDA::Data qw(shape);

# Runs the kernel in the perl interpreter: the kernel source is wrapped in a
# Perl function that loops over every thread index (Fig. 3(b) of the paper).

my @AXES = qw(x y z);
my $SERIAL = 0;

sub new ($class, %args) {
    my $self = bless {%args}, $class;    # dim, size, arguments, retval, source
    $self->{code} = $self->_build;
    return $self;
}

sub name ($self) { 'perl' }

sub perl_source ($self) { $self->{perl_source} }

sub _build ($self) {
    my $dim    = $self->{dim};
    my @axes   = @AXES[ 0 .. $dim - 1 ];
    my $params = join ', ', (map { "\$$_->[0]" } @{ $self->{arguments} }), (map { "\$size_$_" } @axes);
    my $loops  = '';
    my $close  = '';
    for my $axis (@axes) {
        $loops .= "for my \$percuda_$axis (0 .. \$size_$axis - 1) { \$$axis = \$percuda_$axis;\n";
        $close .= '}';
    }
    my $package = 'PerCUDA::Engine::Perl::Kernel' . $SERIAL++;
    my $src = <<"PERL";
package $package;
use strict;
use warnings;
no warnings qw(uninitialized numeric);
sub {
    my ($params) = \@_;
    my (@{[ join ', ', map { "\$$_" } @axes ]});
    my \$percuda_body = sub {
#line 1 "PerCUDA kernel"
$self->{source}
    ;};
    $loops    \$percuda_body->();
    $close
    return;
}
PERL
    $self->{perl_source} = $src;
    my $code = eval $src;    ## no critic (ProhibitStringyEval)
    Carp::croak("PerCUDA: failed to compile the kernel for the perl engine: $@") unless $code;
    return $code;
}

sub run ($self, @values) {
    my @names = map { $_->[0] } @{ $self->{arguments} };
    my %value;
    @value{@names} = @values;
    my $size = $value{ $self->{size} };
    my $dims = shape($size, $self->{dim});
    $self->{code}->(@values, @$dims);
    return map { $value{$_} } @{ $self->{retval} };
}

1;
