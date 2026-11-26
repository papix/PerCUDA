package PerCUDA::Engine::GPU;
use v5.40;
use parent 'PerCUDA::Engine';
use Carp         ();
use Scalar::Util qw(refaddr);
use PerCUDA::Array;
use PerCUDA::CodeGen::CUDA;
use PerCUDA::Data qw(is_ndarray);
use PerCUDA::Compiler;
use PerCUDA::Reduce;

# Runs the kernel on a GPU: the CUDA C++ from PerCUDA::CodeGen::CUDA is
# compiled with NVRTC (cached on disk), arrays are converted and copied to
# reused device buffers by CUDA::DriverAPI, scalars are passed by value, and
# the retval arrays are copied back into the caller's arrays.

my %TYPE  = (i => 'i32', f => 'f32', d => 'f64');
my %SIZE  = (i => 4,     f => 4,     d => 8);
my %BLOCK = (1 => [ 256, 1, 1 ], 2 => [ 32, 8, 1 ], 3 => [ 32, 4, 2 ]);    # CUDA x, y, z

sub init ($self) {

    # context, source, precision, cache_dir
    $self->{params}   = $self->{analyzer}->kernel_params($self->{check});
    $self->{buffers}  = {};    # argument index => CUDA::DriverAPI::Buffer
    $self->{compiler} = PerCUDA::Compiler->new(
        context   => $self->{context},
        precision => $self->{precision},
        cache_dir => $self->{cache_dir},
    );
    $self->{kernel} = $self->_program('kernel', $self->{source});
    return;
}

# A compiled program, from the disk cache when possible:
# { source, program, from_cache }
sub _program ($self, $key, $source) {
    return $self->{programs}{$key} //= do {
        my ($program, $from_cache) = $self->{compiler}->program($source);
        { source => $source, program => $program, from_cache => $from_cache };
    };
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

sub _launch ($self, $slot, @launch) {
    my $ctx = $self->{context};
    return $ctx->run($slot->{program}, @launch) unless $slot->{from_cache};

    # The driver loads a cached image on its first use, so a broken cache
    # entry fails here: compile again and retry once.
    my $ok  = eval { $ctx->run($slot->{program}, @launch); 1 };
    my $err = $@;
    $slot->{from_cache} = 0;
    return if $ok;
    die $err
        unless $err =~
        /CUDA_ERROR_(?:INVALID_IMAGE|INVALID_PTX|INVALID_SOURCE|NO_BINARY_FOR_GPU|UNSUPPORTED_PTX_VERSION)/;
    $slot->{program} = $self->{compiler}->compile($slot->{source});
    $ctx->run($slot->{program}, @launch);
    return;
}

# Reduces a device array (see PerCUDA::Reduce). Returns (value, flat index).
sub reduce ($self, $array, $op) {
    my $type = $array->{type};
    my $n    = 1;
    $n *= $_ for @{ $array->{dims} };
    return ($op eq 'sum' ? 0 : undef, -1) unless $n;

    my $slot   = $self->_program("reduce_${op}_$type", PerCUDA::Reduce::cuda_source($op, $type));
    my $blocks = PerCUDA::Reduce::blocks($n);
    my $acc    = $op eq 'sum' && $type eq 'i' ? 'i64' : $TYPE{$type};
    my $size   = $acc eq 'i64'                ? 8     : $SIZE{$type};
    my ($part, $part_idx) =
        ($self->_buffer('reduce_part', $blocks * $size), $self->_buffer('reduce_part_idx', $blocks * 4));
    my ($result, $result_idx) = ($self->_buffer('reduce_result', $size), $self->_buffer('reduce_result_idx', 4));
    my $threads = $PerCUDA::Reduce::THREADS;
    $self->_launch(
        $slot,
        PerCUDA::Reduce::kernel_name($op, $type, 1),
        [ $array->{buffer}, $part_idx, [ i32 => $n ], $part, $part_idx ],
        [ $blocks, 1, 1, $threads, 1, 1, 0 ]
    );
    $self->_launch(
        $slot,
        PerCUDA::Reduce::kernel_name($op, $type, 2),
        [ $part, $part_idx, [ i32 => $blocks ], $result, $result_idx ],
        [ 1, 1, 1, $threads, 1, 1, 0 ]
    );
    my ($value) = @{ $result->read_array(type => $acc, dims => [1]) };
    my ($index) = @{ $result_idx->read_array(type => 'i32', dims => [1]) };
    return ($value, $index);
}

# Versions of CUDA::DriverAPI before PDL support reject ndarrays as non-array
# references; say what to do instead.
sub _ndarray_guard ($ndarray, $code) {
    return $code->() unless $ndarray;
    my @result = eval { $code->() };
    return @result unless $@;
    Carp::croak("PerCUDA: this CUDA::DriverAPI does not accept PDL ndarrays; update it, or pass nested arrays")
        if $@ =~ /(?:Expected an array reference|into must be an array reference)/;
    die $@;
}

sub make_array ($self, $data, $dims, $rank, $type) {
    my $bytes = $SIZE{$type};
    $bytes *= $_ for @$dims;
    my $buffer = $self->{context}->buffer(size => $bytes || 1);
    if (!defined $data) {
        $buffer->fill(0);
    }
    elsif ($bytes) {
        _ndarray_guard(is_ndarray($data), sub { $buffer->write_array($data, type => $TYPE{$type}, rank => $rank) });
    }
    return PerCUDA::Array->_new(engine => 'gpu', type => $type, dims => $dims, buffer => $buffer, owner => $self);
}

sub run ($self, @values) {
    my $layout = $self->layout(\@values);
    my $grid   = $layout->{grid};

    my (%buf, %shared);
    for my $i (sort { $a <=> $b } keys %{ $layout->{arrays} }) {
        my $arg = $self->{args}[$i];
        my ($dims, $empty) = @{ $layout->{arrays}{$i} }{qw(dims empty)};

        # A device array is already there.
        if ($layout->{arrays}{$i}{device}) {
            $buf{$i} = $values[$i]{buffer};
            next;
        }

        # The same array passed for two read-only arguments is copied once.
        my $key = !$empty && !$arg->{written} ? join(':', refaddr $values[$i], $arg->{type}, $arg->{rank}) : undef;
        if (defined $key && defined $shared{$key}) {
            $buf{$i} = $buf{ $shared{$key} };
            next;
        }
        $shared{$key} = $i if defined $key;

        my $bytes = $SIZE{ $arg->{type} };
        $bytes *= $_ for @$dims;
        $buf{$i} = $self->_buffer($i, $bytes || 1);

        # An array the kernel overwrites entirely needs no upload or zero fill.
        next if $arg->{covered} && "@$dims" eq "@{$grid}";

        if ($empty) {
            $buf{$i}->fill(0);
        }
        elsif ($bytes) {

            # a nested array is converted in C; an ndarray is copied as is
            _ndarray_guard($layout->{arrays}{$i}{ndarray},
                sub { $buf{$i}->write_array($values[$i], type => $TYPE{ $arg->{type} }, rank => $arg->{rank}) });
        }
    }

    my $err;
    if ($self->{check}) {    # flag, array, line, axis, index, x, y, z of the first bad access
        $err = $self->_buffer('check', 8 * 4);
        $err->fill(0);
    }
    my @kernel_args = map {
        my $p = $_;
              $p->{kind} eq 'error'  ? $err
            : $p->{kind} eq 'array'  ? $buf{ $p->{arg} }
            : $p->{kind} eq 'scalar' ? [ $TYPE{ $p->{type} } => $values[ $p->{arg} ] ]
            : $p->{kind} eq 'size'   ? [ i32                 => $grid->[ $p->{axis} ] ]
            :                          [ i32 => $layout->{arrays}{ $p->{arg} }{dims}[ $p->{axis} ] ];
    } @{ $self->{params} };

    # the last Perl axis runs along CUDA's x axis (see PerCUDA::CodeGen::CUDA)
    my $dim    = $self->{dim};
    my @block  = @{ $BLOCK{$dim} };
    my @cuda   = map { $_ < $dim ? $grid->[ $dim - 1 - $_ ] : 1 } 0 .. 2;
    my @blocks = map { int(($cuda[$_] + $block[$_] - 1) / $block[$_]) } 0 .. 2;
    $self->_launch($self->{kernel}, $PerCUDA::CodeGen::CUDA::KERNEL_NAME, \@kernel_args, [ @blocks, @block, 0 ])
        unless grep { $_ == 0 } @blocks;

    # check => 1: report a bad access before anything is copied back
    if ($err) {
        my ($bad, $id, $line, $axis, $index, @thread) = @{ $err->read_array(type => 'i32', dims => [8]) };
        if ($bad) {
            my $arg = $self->{args}[$id];
            PerCUDA::Engine::out_of_range($arg->{name}, $line, [ @thread[ 0 .. $dim - 1 ] ],
                $axis, $index, $layout->{arrays}{$id}{dims}[$axis]);
        }
    }

    my @results;
    for my $name (@{ $self->{retval} }) {
        my $i = $self->{index}{$name};

        # A device array stays on the device; run returns without waiting for
        # the kernel, and errors surface at the next copy back.
        if ($layout->{arrays}{$i}{device}) {
            push @results, $values[$i];
            next;
        }

        # filled in place: a nested array reuses its rows and scalars, an
        # ndarray (checked to have the matching type) is reshaped
        _ndarray_guard(
            $layout->{arrays}{$i}{ndarray},
            sub {
                $buf{$i}->read_array(
                    type => $TYPE{ $self->{args}[$i]{type} },
                    dims => $layout->{arrays}{$i}{dims},
                    into => $values[$i],
                );
            }
        );
        push @results, $values[$i];
    }
    $self->{context}->synchronize unless @results;    # report kernel errors even without retval
    return @results;
}

1;
