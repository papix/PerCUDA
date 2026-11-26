use v5.40;
use Test2::V0;
use File::Temp ();
use lib 't/lib';
use PerCUDATest;
use PerCUDA;

skip_all 'no GPU' unless gpu_available();

my $dir  = File::Temp->newdir;
my %args = (
    engine    => 'gpu',
    cache_dir => "$dir",
    argument  => [ in => 'f', out => 'f' ],
    retval    => ['out'],
    source    => 'my $x = thread_index(); $out->[$x] = $in->[$x] * 2;',
);

sub cache_files () { glob "$dir/*.bin" }

is scalar PerCUDA->new(%args)->run([ 1 .. 4 ], []), [ 2, 4, 6, 8 ], 'first run compiles';
my @files = cache_files();
is scalar @files, 1, 'compiled image is cached';

my $r = PerCUDA->new(%args);
ok $r->{engine}{kernel}{from_cache}, 'second object loads the cache';
is scalar $r->run([ 1 .. 4 ], []), [ 2, 4, 6, 8 ], 'cached image runs';

# The driver loads images lazily, so a broken cache fails on the first run.
open my $fh, '>:raw', $files[0] or die $!;
print {$fh} "not a cubin";
close $fh;
my $broken = PerCUDA->new(%args);
ok $broken->{engine}{kernel}{from_cache}, 'broken cache is loaded without an error';
is scalar $broken->run([ 1 .. 4 ], []), [ 2, 4, 6, 8 ], 'run recompiles when the cached image is invalid';
ok -s $files[0] > length "not a cubin", 'cache entry is replaced';

done_testing;
