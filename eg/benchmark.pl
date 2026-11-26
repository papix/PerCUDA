# Usage: perl examples/benchmark.pl [gpu|perl] [n]
# Times n x n matrix multiplication (the experiment of Table 1 in the paper).
use v5.40;
use FindBin;
use lib "$FindBin::Bin/../lib";
use Time::HiRes qw(time);
use PerCUDA     qw(:kernel configure runner);

my ($engine, $n) = @ARGV;
$engine //= 'gpu';
$n      //= 1024;

# n is an int (a loop bound); the arrays are float.
sub matmul : GPU ($a, $b, $c, $n) {
    my $sum = 0.0;
    for (my $k = 0; $k < $n; $k++) {
        $sum += $a->[$x][$k] * $b->[$k][$y];
    }
    $c->[$x][$y] = $sum;
}

configure(engine => $engine);

my $a = [
    map {
        my $i = $_;
        [ map { ($i + $_) % 7 } 0 .. $n - 1 ]
    } 0 .. $n - 1
];
my $b = [
    map {
        my $i = $_;
        [ map { ($i * $_) % 5 } 0 .. $n - 1 ]
    } 0 .. $n - 1
];

for my $round (1 .. ($engine eq 'gpu' ? 3 : 1)) {
    my $start = time;
    my $c     = matmul($a, $b, [], $n);
    printf "engine=%s n=%d round=%d: %.4fs (c[1][2]=%s)\n", runner(\&matmul)->engine, $n, $round,
        time - $start, $c->[1][2];
}
