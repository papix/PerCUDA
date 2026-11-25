use v5.40;
use Test2::V0;
use lib 't/lib';
use PerCUDATest;
use PerCUDA;

skip_all 'no GPU' unless gpu_available();

# The set boundary amplifies any rounding difference, so this checks that
# both engines round every operation the same way.
my $KERNEL = <<'K';
my $cr = -2.0 + 3.0 * $x / $size_x;
my $ci = -1.5 + 3.0 * $y / $size_y;
my $zr = 0.0;
my $zi = 0.0;
my $n = 0;
while ($n < $max && $zr * $zr + $zi * $zi <= 4.0) {
    my $t = $zr * $zr - $zi * $zi + $cr;
    $zi = 2.0 * $zr * $zi + $ci;
    $zr = $t;
    $n++;
}
$out->[$x][$y] = $n;
K

my $W = 64;
my %result;
for my $precision (qw(single double)) {
    for my $engine (qw(perl gpu)) {
        my $r = PerCUDA->new(
            dim       => 2,
            engine    => $engine,
            precision => $precision,
            argument  => [ out => 'i', max => 'i' ],
            size      => 'out',
            retval    => ['out'],
            source    => $KERNEL,
        );
        $result{$precision}{$engine} = $r->run([ map { [ (0) x $W ] } 1 .. $W ], 256);
    }
    is $result{$precision}{gpu}, $result{$precision}{perl}, "$precision: perl and gpu agree on every pixel";
}

my $gpu = $result{single}{gpu};
is $gpu->[ $W * 2 / 3 ][ $W / 2 ], 256, 'origin is in the set';
is $gpu->[0][0],                   1,   'corner escapes immediately';
isnt join(',', map { @$_ } @$gpu), join(',', map { @$_ } @{ $result{double}{gpu} }),
    'single and double precision differ';

done_testing;
