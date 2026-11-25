package PerCUDA::Compiler;
use v5.40;
use Carp ();
use Digest::MD5 qw(md5_hex);
use File::Path qw(make_path);
use File::Spec;
use File::Temp ();

# Compiles generated C to PTX with clang's NVPTX backend and caches the result,
# so an unchanged kernel is compiled only once.

sub new ($class, %args) {
    return bless {
        clang     => $args{clang} // $ENV{PERCUDA_CLANG} // 'clang',
        arch      => $args{arch} // $ENV{PERCUDA_ARCH} // 'sm_75',
        cache_dir => $args{cache_dir} // $ENV{PERCUDA_CACHE_DIR} // default_cache_dir(),
    }, $class;
}

sub default_cache_dir () {
    my $base = $ENV{XDG_CACHE_HOME} // File::Spec->catdir($ENV{HOME} // File::Spec->tmpdir, '.cache');
    return File::Spec->catdir($base, 'percuda');
}

sub available ($self) {
    my $clang = $self->{clang};
    return -x $clang if File::Spec->file_name_is_absolute($clang);
    for my $dir (File::Spec->path) {
        return 1 if -x File::Spec->catfile($dir, $clang);
    }
    return 0;
}

sub compile ($self, $c_source) {
    my @flags = ('--target=nvptx64-nvidia-cuda', "-march=$self->{arch}", '-O3', '-ffp-contract=off', '-S');
    my $key   = md5_hex(join "\0", $c_source, @flags);
    my $ptx   = File::Spec->catfile($self->{cache_dir}, "$key.ptx");
    return $ptx if -s $ptx;

    make_path($self->{cache_dir});
    my $c_file = File::Spec->catfile($self->{cache_dir}, "$key.c");
    _write($c_file, $c_source);

    my $tmp = File::Temp->new(DIR => $self->{cache_dir}, SUFFIX => '.ptx', UNLINK => 1);
    close $tmp;
    my $status = system { $self->{clang} } $self->{clang}, @flags, '-o', $tmp->filename, $c_file;
    if ($status != 0) {
        Carp::croak("PerCUDA: clang failed to compile the kernel (exit $status). Generated C: $c_file");
    }
    rename $tmp->filename, $ptx or Carp::croak("PerCUDA: cannot write $ptx: $!");
    $tmp->unlink_on_destroy(0);
    return $ptx;
}

sub _write ($path, $content) {
    open my $fh, '>', $path or Carp::croak("PerCUDA: cannot write $path: $!");
    print {$fh} $content;
    close $fh or Carp::croak("PerCUDA: cannot write $path: $!");
}

1;
