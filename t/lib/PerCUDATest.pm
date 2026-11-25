package PerCUDATest;
use v5.40;
use Exporter 'import';
use File::Temp ();
use PerCUDA::Engine::GPU;

our @EXPORT = qw(gpu_available both_engines ndarray_engines);

my $CACHE = File::Temp->newdir;
$ENV{PERCUDA_CACHE_DIR} = "$CACHE";

sub gpu_available () {
    state $ok = PerCUDA::Engine::GPU->context_available ? 1 : 0;
    return $ok;
}

# Engines to test: perl always, gpu when available.
sub both_engines () { return gpu_available() ? qw(perl gpu) : qw(perl) }

# Engines that accept PDL ndarrays: the gpu engine needs a CUDA::DriverAPI
# with PDL support. Call only after loading PDL.
sub ndarray_engines () {
    state $gpu = gpu_available() && eval {
        my $buffer = PerCUDA::Engine::GPU->context_available->buffer(size => 4);
        $buffer->write_array(PDL->pdl([1]), type => 'f32', rank => 1);
        1;
    };
    return $gpu ? qw(perl gpu) : qw(perl);
}

1;
