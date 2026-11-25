package PerCUDA::Engine::GPU;
use v5.40;
use Carp ();
use PerCUDA::CodeGen;
use PerCUDA::Data qw(shape flatten unflatten);

# Runs the kernel on a GPU: Perl data is packed into binary, copied to the
# device, the generated PTX is launched through CUDA::DriverAPI, and the
# return values are copied back and converted to Perl arrays.

my %PACK = (i => 'l', f => 'f', d => 'd');
my %SIZE = (i => 4, f => 4, d => 8);
my %BLOCK = (1 => [ 256, 1, 1 ], 2 => [ 16, 16, 1 ], 3 => [ 8, 8, 4 ]);

sub new ($class, %args) {
    my $self = bless {%args}, $class;    # dim, size, retval, args, ptx, context
    return $self;
}

sub name ($self) { 'gpu' }

sub context_available ($class, $device = 0) {
    return eval {
        require CUDA::DriverAPI;
        CUDA::DriverAPI->new(device => $device);
    };
}

sub run ($self, @values) {
    my $ctx  = $self->{context};
    my @args = @{ $self->{args} };
    my %index = map { $args[$_]{name} => $_ } 0 .. $#args;
    my %is_retval = map { $_ => 1 } @{ $self->{retval} };

    my $size_arg  = $args[ $index{ $self->{size} } ];
    my $size_rank = $size_arg->{rank} || $self->{dim};
    my $size_dims = shape($values[ $index{ $self->{size} } ], $size_rank);
    Carp::croak("PerCUDA: size argument '$self->{size}' has fewer than $self->{dim} dimensions")
        if @$size_dims < $self->{dim};
    my @grid_size = (@{$size_dims}[ 0 .. $self->{dim} - 1 ], 1, 1)[ 0 .. 2 ];

    my @dims_buf = @grid_size;
    my @buffers;    # [ bytes, dims ] per argument
    for my $i (0 .. $#args) {
        my $arg   = $args[$i];
        my $value = $values[$i];
        my $fmt   = $PACK{ $arg->{type} } . '*';
        if ($arg->{kind} eq 'array') {
            Carp::croak("PerCUDA: argument '$arg->{name}' must be an array reference")
                unless ref $value eq 'ARRAY';
            my ($flat, $dims);
            if (@$value) {
                ($flat, $dims) = flatten($value, $arg->{rank});
            }
            else {
                # an empty array (typically an output) takes the shape of the size argument
                Carp::croak("PerCUDA: cannot infer the shape of empty array '$arg->{name}'")
                    if $arg->{rank} > @$size_dims;
                $dims = [ @{$size_dims}[ 0 .. $arg->{rank} - 1 ] ];
                my $n = 1;
                $n *= $_ for @$dims;
                $flat = [ (0) x $n ];
            }
            no warnings 'uninitialized';
            push @buffers, [ pack($fmt, @$flat), $dims ];
            push @dims_buf, (@$dims, 0, 0)[ 0 .. 2 ];
        }
        elsif ($arg->{kind} eq 'scalar') {
            Carp::croak("PerCUDA: argument '$arg->{name}' must be a number") if ref $value;
            no warnings 'uninitialized';
            push @buffers, [ pack($fmt, $value), undef ];
            push @dims_buf, 0, 0, 0;
        }
        else {
            push @buffers, [ "\0" x 8, undef ];
            push @dims_buf, 0, 0, 0;
        }
    }
    push @buffers, [ pack('l*', @dims_buf), undef ];

    my @ptrs;
    my @results;
    my $ok = eval {
        for my $buf (@buffers) {
            my $bytes = length $buf->[0] ? $buf->[0] : "\0" x 8;
            my $ptr = $ctx->malloc($bytes);
            push @ptrs, $ptr;
            $ctx->transfer_h2d($bytes, $ptr);
        }

        my @block = @{ $BLOCK{ $self->{dim} } };
        my @grid  = map { $grid_size[$_] ? int(($grid_size[$_] + $block[$_] - 1) / $block[$_]) : 0 } 0 .. 2;
        if (!grep { $_ == 0 } @grid) {
            $ctx->run($self->{ptx}, $PerCUDA::CodeGen::KERNEL_NAME, \@ptrs, [ @grid, @block, 0 ]);
        }

        for my $name (@{ $self->{retval} }) {
            my $i   = $index{$name};
            my $arg = $args[$i];
            my $out = "\0" x length $buffers[$i][0];
            $ctx->transfer_d2h($ptrs[$i], \$out) if length $out;
            my @flat   = unpack $PACK{ $arg->{type} } . '*', $out;
            my $result = unflatten(\@flat, $buffers[$i][1]);
            @{ $values[$i] } = @$result;    # update the caller's array like the perl engine does
            push @results, $values[$i];
        }
        1;
    };
    my $err = $@;
    $ctx->free($_) for @ptrs;
    die $err unless $ok;
    return @results;
}

1;
