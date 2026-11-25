use v5.40;
use Test::More;
use lib 't/lib';
use PerCUDATest;
use PerCUDA;

plan skip_all => 'no GPU' unless gpu_available();

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
for my $engine (qw(perl gpu)) {
    my $r = PerCUDA->new(
        dim      => 2,
        engine   => $engine,
        argument => [ out => 'i', max => 'i' ],
        size     => 'out',
        retval   => ['out'],
        source   => $KERNEL,
    );
    my $grid = [ map { [ (0) x $W ] } 1 .. $W ];
    $result{$engine} = $r->run($grid, 256);
}

# float on the GPU vs double in perl: pixels on the set boundary may differ
my ($same, $total) = (0, 0);
for my $x (0 .. $W - 1) {
    for my $y (0 .. $W - 1) {
        $total++;
        $same++ if $result{perl}[$x][$y] == $result{gpu}[$x][$y];
    }
}
cmp_ok $same / $total, '>=', 0.97, "perl and gpu agree on $same/$total pixels";
is $result{gpu}[ $W * 2 / 3 ][ $W / 2 ], 256, 'origin is in the set';
is $result{gpu}[0][0], 1, 'corner escapes immediately';

my $double = PerCUDA->new(
    dim       => 2,
    engine    => 'gpu',
    precision => 'double',
    argument  => [ out => 'i', max => 'i' ],
    size      => 'out',
    retval    => ['out'],
    source    => $KERNEL,
);
is_deeply scalar $double->run([ map { [ (0) x $W ] } 1 .. $W ], 256), $result{perl},
    'precision => double matches perl on every pixel';

done_testing;
