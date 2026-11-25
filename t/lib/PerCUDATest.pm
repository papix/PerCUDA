package PerCUDATest;
use v5.40;
use Exporter 'import';
use File::Temp ();
use PerCUDA::Engine::GPU;

our @EXPORT = qw(gpu_available both_engines);

my $CACHE = File::Temp->newdir;
$ENV{PERCUDA_CACHE_DIR} = "$CACHE";

sub gpu_available () {
    state $ok = PerCUDA::Engine::GPU->context_available ? 1 : 0;
    return $ok;
}

# Engines to test: perl always, gpu when available.
sub both_engines () { gpu_available() ? qw(perl gpu) : qw(perl) }

1;
