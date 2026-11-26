package PerCUDA::Context;
use v5.40;
use Carp    ();
use PerCUDA ();
use PerCUDA::Engine::GPU;
use PerCUDA::Engine::Perl;

our @CARP_NOT = qw(PerCUDA PerCUDA::Engine);

# Options shared by several kernels (PerCUDA->context): the engine is chosen
# once, and device arrays can be made without a kernel.

my %OPTION =
    map { $_ => 1 } qw(argument size retval constant precision mode engine check perl_jobs cache_dir device);

sub new ($class, %defaults) {
    for my $key (sort keys %defaults) {
        Carp::croak("PerCUDA: unknown option '$key'") unless $OPTION{$key};
    }
    my $engine = $defaults{engine} // PerCUDA::default_engine();
    Carp::croak("PerCUDA: engine must be 'gpu' or 'perl'") if defined $engine && $engine !~ /\A(?:gpu|perl)\z/;
    Carp::croak("PerCUDA: precision must be 'single' or 'double'")
        if defined $defaults{precision} && $defaults{precision} !~ /\A(?:single|double)\z/;
    Carp::croak("PerCUDA: mode must be 'fast', 'compat' or 'exact'")
        if defined $defaults{mode} && $defaults{mode} !~ /\A(?:fast|compat|exact)\z/;
    my $context;
    if (!defined $engine || $engine eq 'gpu') {
        $context = PerCUDA::Engine::GPU->context_available($defaults{device} // 0);
        Carp::croak("PerCUDA: no usable GPU (CUDA::DriverAPI could not create a context)")
            if !$context && defined $engine;
    }
    $defaults{engine} = $context ? 'gpu' : 'perl';
    my $storage =
        $context
        ? PerCUDA::Engine::GPU->storage(
        context   => $context,
        precision => $defaults{precision} // 'single',
        cache_dir => $defaults{cache_dir},
        )
        : PerCUDA::Engine::Perl->storage(mode => $defaults{mode} // 'compat');
    return bless { defaults => \%defaults, storage => $storage }, $class;
}

sub engine ($self) { return $self->{defaults}{engine} }

sub kernel ($self, %args) {
    for my $name (qw(engine device)) {
        Carp::croak("PerCUDA: '$name' is fixed by the context") if exists $args{$name};
    }
    Carp::croak("PerCUDA: 'constant' must be a hash reference of name => number")
        if defined $args{constant} && ref $args{constant} ne 'HASH';

    # PerCUDA->new merges: context < :GPU attribute < %args (engine and
    # device stay the context's).
    return PerCUDA->new(%args, _context => $self->{defaults});
}

sub array ($self, @args) { return PerCUDA::_make_array($self->{storage}, $self->{defaults}{precision}, @args) }

1;
__END__

=encoding utf-8

=head1 NAME

PerCUDA::Context - Options shared by several kernels

=head1 SYNOPSIS

    use PerCUDA;

    sub step ($u, $v) { $v->[$x][$y] = 0.5 * $u->[$x][$y] }

    my $pc   = PerCUDA->context(engine => $ARGV[0], precision => 'double');
    my $step = $pc->kernel(source => \&step);
    my $u    = $pc->array(dims => [ 512, 512 ]);    # double

=head1 DESCRIPTION

Returned by C<< PerCUDA->context(%options) >>. The options are those of
C<< PerCUDA->new >>; they become the defaults of every kernel made by the
context. The engine is chosen once, when the context is made, so all its
kernels and arrays are on the same engine.

=head1 METHODS

=over 4

=item C<< $context->kernel(%args) >>

C<< PerCUDA->new(%options, %args) >>. Any option but C<engine> and C<device>
can be overridden. The C<constant> of C<%args> adds to that of the context
(a name in both takes the kernel's value).

=item C<< $context->array(...) >>

A L<PerCUDA::Array> on the context's engine, with the arguments of
C<< $runner->array >>. Any kernel of the context accepts it.

=item C<< $context->engine >>

C<gpu> or C<perl>.

=back

=cut
