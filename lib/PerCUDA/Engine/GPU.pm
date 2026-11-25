package PerCUDA::Engine::GPU;
use v5.40;
use parent 'PerCUDA::Engine';
use Carp ();
use PerCUDA::CodeGen::CUDA;
use PerCUDA::Compiler;

# Runs the kernel on a GPU: the CUDA C++ from PerCUDA::CodeGen::CUDA is
# compiled with NVRTC (cached on disk), arrays are converted and copied to
# reused device buffers by CUDA::DriverAPI, scalars are passed by value, and
# the retval arrays are copied back into the caller's arrays.

my %TYPE  = (i => 'i32', f => 'f32', d => 'f64');
my %SIZE  = (i => 4,     f => 4,     d => 8);
my %BLOCK = (1 => [ 256, 1, 1 ], 2 => [ 32, 8, 1 ], 3 => [ 32, 4, 2 ]);    # CUDA x, y, z

sub init ($self) {

    # context, source, precision, cache_dir
    $self->{params}   = $self->{analyzer}->kernel_params;
    $self->{buffers}  = {};    # argument index => CUDA::DriverAPI::Buffer
    $self->{compiler} = PerCUDA::Compiler->new(
        context   => $self->{context},
        precision => $self->{precision},
        cache_dir => $self->{cache_dir},
    );
    @$self{qw(program from_cache)} = $self->{compiler}->program($self->{source});
    return;
}

sub name ($self) { return 'gpu' }

# One context per device, shared by every runner for the life of the process:
# releasing the last reference destroys the primary context, and creating it
# again costs tens of milliseconds.
my %CONTEXT;

sub context_available ($class, $device = 0) {
    return $CONTEXT{$device} //= eval {
        require CUDA::DriverAPI;
        CUDA::DriverAPI->new(device => $device);
    };
}

# Device buffer of argument $i, reused while its size does not change.
sub _buffer ($self, $i, $size) {
    my $buf = $self->{buffers}{$i};
    return $buf if $buf && $buf->size == $size;
    undef $self->{buffers}{$i};    # free the old one first
    return $self->{buffers}{$i} = $self->{context}->buffer(size => $size);
}

sub _launch ($self, @launch) {
    my $ctx = $self->{context};
    return $ctx->run($self->{program}, @launch) unless $self->{from_cache};

    # The driver loads a cached image on its first use, so a broken cache
    # entry fails here: compile again and retry once.
    my $ok  = eval { $ctx->run($self->{program}, @launch); 1 };
    my $err = $@;
    $self->{from_cache} = 0;
    return if $ok;
    die $err
        unless $err =~
        /CUDA_ERROR_(?:INVALID_IMAGE|INVALID_PTX|INVALID_SOURCE|NO_BINARY_FOR_GPU|UNSUPPORTED_PTX_VERSION)/;
    $self->{program} = $self->{compiler}->compile($self->{source});
    $ctx->run($self->{program}, @launch);
    return;
}

sub run ($self, @values) {
    my $layout = $self->layout(\@values);
    my $grid   = $layout->{grid};

    my %buf;
    for my $i (keys %{ $layout->{arrays} }) {
        my $arg = $self->{args}[$i];
        my ($dims, $empty) = @{ $layout->{arrays}{$i} }{qw(dims empty)};
        my $bytes = $SIZE{ $arg->{type} };
        $bytes *= $_ for @$dims;
        $buf{$i} = $self->_buffer($i, $bytes || 1);
        if ($empty) {
            $buf{$i}->fill(0);
        }
        elsif ($bytes) {
            $buf{$i}->write_array($values[$i], type => $TYPE{ $arg->{type} }, rank => $arg->{rank});
        }
    }

    my @kernel_args = map {
        my $p = $_;
              $p->{kind} eq 'array'  ? $buf{ $p->{arg} }
            : $p->{kind} eq 'scalar' ? [ $TYPE{ $p->{type} } => $values[ $p->{arg} ] ]
            : $p->{kind} eq 'size'   ? [ i32                 => $grid->[ $p->{axis} ] ]
            :                          [ i32 => $layout->{arrays}{ $p->{arg} }{dims}[ $p->{axis} ] ];
    } @{ $self->{params} };

    # the last Perl axis runs along CUDA's x axis (see PerCUDA::CodeGen::CUDA)
    my $dim    = $self->{dim};
    my @block  = @{ $BLOCK{$dim} };
    my @cuda   = map { $_ < $dim ? $grid->[ $dim - 1 - $_ ] : 1 } 0 .. 2;
    my @blocks = map { int(($cuda[$_] + $block[$_] - 1) / $block[$_]) } 0 .. 2;
    $self->_launch($PerCUDA::CodeGen::CUDA::KERNEL_NAME, \@kernel_args, [ @blocks, @block, 0 ])
        unless grep { $_ == 0 } @blocks;

    my @results;
    for my $name (@{ $self->{retval} }) {
        my $i = $self->{index}{$name};

        # filled in place, reusing the caller's rows and scalars
        $buf{$i}->read_array(
            type => $TYPE{ $self->{args}[$i]{type} },
            dims => $layout->{arrays}{$i}{dims},
            into => $values[$i],
        );
        push @results, $values[$i];
    }
    $self->{context}->synchronize unless @results;    # report kernel errors even without retval
    return @results;
}

1;
