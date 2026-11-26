# Monte Carlo pricing of a European call under geometric Brownian motion
# (S0 = 100, strike 100, r = 5%, sigma = 20%, T = 1 year, the textbook case
# whose Black-Scholes price is 10.4506). Each path draws one standard normal
# with the Box-Muller transform; the price is the discounted mean payoff,
# with its standard error from the sum of squares (both reductions). The
# check is that the estimate lies within 4 standard errors of the exact price.
#
#   perl eg/mc-option.pl [gpu|perl] [paths] [seed]
use v5.40;
use FindBin;
use lib "$FindBin::Bin/../lib";
use Time::HiRes qw(time);
use PerCUDA;

my ($engine, $paths, $seed) = @ARGV;
$paths //= 50_000_000;
$seed  //= 1;
my ($s0, $strike, $rate, $sigma, $expiry) = (100, 100, 0.05, 0.2, 1);
my $exact = 10.450583572185565;

my $simulate = PerCUDA->new(
    engine    => $engine,
    precision => 'double',
    argument  => [ payoff => 'd', square => 'd', seed => 'i', s0 => 'd', k => 'd', r => 'd', sigma => 'd', t => 'd' ],
    source    => <<'KERNEL');
my $u1 = 1.0 - random($seed, $x, 0);    # (0, 1]: log(0) would be -inf
my $u2 = random($seed, $x, 1);
my $z = sqrt(-2.0 * log($u1)) * cos(6.283185307179586 * $u2);
my $st = $s0 * exp(($r - 0.5 * $sigma * $sigma) * $t + $sigma * sqrt($t) * $z);
my $pay = $st > $k ? $st - $k : 0.0;
$payoff->[$x] = $pay;
$square->[$x] = $pay * $pay;
KERNEL

my $start = time;
my ($payoff, $square) = map { $simulate->array(type => 'd', dims => [$paths]) } 1 .. 2;
$simulate->run($payoff, $square, $seed, $s0, $strike, $rate, $sigma, $expiry);
my $mean     = $payoff->sum / $paths;
my $variance = $square->sum / $paths - $mean**2;
my $discount = exp(-$rate * $expiry);
my $price    = $discount * $mean;
my $error    = $discount * sqrt($variance / $paths);
my $elapsed  = time - $start;

printf "engine: %s\n%d paths in %.3f s\n",                         $simulate->engine, $paths, $elapsed;
printf "price %.5f +- %.5f (exact %.5f, %+.2f standard errors)\n", $price, $error, $exact, ($price - $exact) / $error;
say 'check: ', abs($price - $exact) < 4 * $error ? 'ok' : 'NG';
