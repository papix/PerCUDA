# k-means clustering (Lloyd's algorithm) of 2D points drawn from K Gaussian
# blobs (sigma 1, centres on a circle of radius 6). Each iteration assigns
# every point to its nearest centre on the device, then moves each centre to
# the mean of its points with masked sums (reductions). It stops when no
# label changes. The check matches the found centres with the true ones.
#
#   perl eg/kmeans.pl [gpu|perl] [points] [K] [seed]
use v5.40;
use FindBin;
use lib "$FindBin::Bin/../lib";
use Time::HiRes qw(time);
use PerCUDA;

my ($engine, $n, $k, $seed) = @ARGV;
$n    //= 2_000_000;
$k    //= 6;
$seed //= 1;
my @truth = map { [ 6 * cos(2 * 3.14159265358979 * $_ / $k), 6 * sin(2 * 3.14159265358979 * $_ / $k) ] } 0 .. $k - 1;

my %common   = (engine => $engine, size => 'px');
my $generate = PerCUDA->new(
    %common,
    argument => [ px => 'f', py => 'f', centre => 'f', k => 'i', seed => 'i' ],
    source   => <<'KERNEL');
my $c = $x % $k;
my $r = sqrt(-2.0 * log(1.0 - random($seed, $x, 0)));
my $phi = 6.283185307179586 * random($seed, $x, 1);
$px->[$x] = $centre->[$c][0] + $r * cos($phi);
$py->[$x] = $centre->[$c][1] + $r * sin($phi);
KERNEL
my $assign = PerCUDA->new(
    %common,
    argument => [ px => 'f', py => 'f', centre => 'f', k => 'i', label => 'i', changed => 'i' ],
    source   => <<'KERNEL');
my $best = 0;
my $best_d = 3.4e38;
for my $c (0 .. $k - 1) {
    my $dx = $px->[$x] - $centre->[$c][0];
    my $dy = $py->[$x] - $centre->[$c][1];
    my $d = $dx * $dx + $dy * $dy;
    if ($d < $best_d) {
        $best_d = $d;
        $best = $c;
    }
}
$changed->[$x] = $label->[$x] != $best ? 1 : 0;
$label->[$x] = $best;
KERNEL
my $mask = PerCUDA->new(
    %common,
    argument => [ px => 'f', py => 'f', label => 'i', c => 'i', wx => 'f', wy => 'f', w => 'i' ],
    source   => <<'KERNEL');
my $in = $label->[$x] == $c;
$wx->[$x] = $in ? $px->[$x] : 0.0;
$wy->[$x] = $in ? $py->[$x] : 0.0;
$w->[$x] = $in ? 1 : 0;
KERNEL

my $start = time;
my ($px, $py, $wx, $wy) = map { $assign->array(type => 'f', dims => [$n]) } 1 .. 4;
my ($label, $changed, $w) = map { $assign->array(type => 'i', dims => [$n]) } 1 .. 3;
$generate->run($px, $py, \@truth, $k, $seed);

# start from K points spread through the data (k-means++ would be better)
my ($all_x, $all_y) = ($px->get, $py->get);
my $centres    = [ map { my $i = int($_ * $n / $k + $_); [ $all_x->[$i], $all_y->[$i] ] } 0 .. $k - 1 ];
my $iterations = 0;
while ($iterations < 100) {
    $iterations++;
    $assign->run($px, $py, $centres, $k, $label, $changed);
    last if $changed->sum == 0 && $iterations > 1;
    for my $c (0 .. $k - 1) {
        $mask->run($px, $py, $label, $c, $wx, $wy, $w);
        my $count = $w->sum or next;
        $centres->[$c] = [ $wx->sum / $count, $wy->sum / $count ];
    }
}
my $elapsed = time - $start;

my $worst = 0;
for my $t (@truth) {
    my ($nearest) = sort { $a <=> $b } map { sqrt(($_->[0] - $t->[0])**2 + ($_->[1] - $t->[1])**2) } @$centres;
    $worst = $nearest if $nearest > $worst;
}
printf "engine: %s\n%d points, K = %d, %d iterations in %.3f s\n", $assign->engine, $n, $k, $iterations, $elapsed;
printf "centres: %s\n", join ' ', map { sprintf '(%.2f, %.2f)', @$_ } @$centres;
printf "largest distance from a true centre to the nearest found one: %.4f\n", $worst;
say 'check: ', $worst < 10 / sqrt($n / $k) + 0.02 ? 'ok' : 'NG';
