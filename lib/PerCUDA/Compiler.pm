package PerCUDA::Compiler;
use v5.40;
use Carp ();
use Digest::MD5 qw(md5_hex);
use File::Path qw(make_path);
use File::Spec;
use File::Temp ();

# Compiles generated CUDA C++ with NVRTC (through CUDA::DriverAPI) for the
# device's own architecture, and caches the compiled image on disk keyed by
# the source, the compute capability and the options.

sub new ($class, %args) {
    my $context = $args{context} // Carp::croak("context is required");
    my $self = bless {
        context   => $context,
        cache_dir => $args{cache_dir} // $ENV{PERCUDA_CACHE_DIR} // default_cache_dir(),
        # no fused multiply-add, so that every operation rounds like the perl engine
        options => ['--fmad=false'],
    }, $class;
    $self->{device} = join '.', $context->compute_capability;
    return $self;
}

sub default_cache_dir () {
    my $base = $ENV{XDG_CACHE_HOME} // File::Spec->catdir($ENV{HOME} // File::Spec->tmpdir, '.cache');
    return File::Spec->catdir($base, 'percuda');
}

sub cache_path ($self, $source) {
    my $key = md5_hex(join "\0", $source, $self->{device}, @{ $self->{options} }, $CUDA::DriverAPI::VERSION);
    return File::Spec->catfile($self->{cache_dir}, "$key.bin");
}

# Returns ($program, $from_cache). The driver loads images lazily, so a broken
# cache entry only shows up on the first run; callers then use compile().
sub program ($self, $source) {
    my $path = $self->cache_path($source);
    if (-s $path && open my $fh, '<:raw', $path) {
        my $image = do { local $/; <$fh> };
        return ($self->{context}->load($image), 1) if length $image;
    }
    return ($self->compile($source), 0);
}

# Compiles with NVRTC and replaces the cache entry.
sub compile ($self, $source) {
    my $program = $self->{context}->compile($source, name => 'percuda.cu', options => $self->{options});
    my $path    = $self->cache_path($source);
    eval {
        make_path($self->{cache_dir});
        my $tmp = File::Temp->new(DIR => $self->{cache_dir}, SUFFIX => '.tmp');
        binmode $tmp;
        print {$tmp} $program->image;
        close $tmp or die "close: $!";
        rename $tmp->filename, $path or die "rename: $!";
        $tmp->unlink_on_destroy(0);
        1;
    } or Carp::carp("PerCUDA: cannot write the kernel cache $path: $@");
    return $program;
}

1;
