# Estimates pi by Monte Carlo: random points in the unit square, counting
# those inside the quarter circle. Shows random() and reductions.
#
#   perl eg/pi.pl [gpu|perl] [samples] [seed]
use v5.40;
use FindBin;
use lib "$FindBin::Bin/../lib";
use Time::HiRes qw(time);
use PerCUDA     qw(:kernel configure array runner);

my ($engine, $samples, $seed) = @ARGV;
$samples //= 10_000_000;
$seed    //= 1;

# The hit counts are integers; seed is an int because random() takes it.
sub inside : GPU({ argument => [ hit => 'int' ] }) ($hit, $seed) {
    my $u = random($seed, $x, 0);
    my $v = random($seed, $x, 1);
    $hit->[$x] = $u * $u + $v * $v < 1.0 ? 1 : 0;
}

configure(engine => $engine);

my $start = time;
my $hit   = array(type => 'int', dims => [$samples]);
inside($hit, $seed);
my $pi = 4 * $hit->sum / $samples;
printf "engine: %s\npi = %.6f (error %.6f), %d samples in %.3f s\n", runner(\&inside)->engine, $pi,
    $pi - 3.14159265358979,
    $samples, time - $start;
