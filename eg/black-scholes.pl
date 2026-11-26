# European option pricing with the Black-Scholes formula, with the inputs
# and the cumulative normal approximation (Abramowitz & Stegun 26.2.17) of
# NVIDIA's CUDA samples: stock price 5..30, strike 1..100, expiry 0.25..10
# years, riskless rate 2%, volatility 30%. The check compares a few options
# with a double precision Perl implementation and verifies put-call parity
# C - P = S - X exp(-rT) on every option.
#
#   perl eg/black-scholes.pl [gpu|perl] [options] [seed]
use v5.40;
use FindBin;
use lib "$FindBin::Bin/../lib";
use Time::HiRes qw(time);
use PerCUDA     qw(:kernel configure array runner);

my ($engine, $n, $seed) = @ARGV;
$n    //= 8_000_000;
$seed //= 1;
my ($rate, $volatility) = (0.02, 0.30);

sub inputs : GPU ($s, $k, $t, $seed) {
    $s->[$x] = 5.0 + 25.0 * random($seed, $x, 0);
    $k->[$x] = 1.0 + 99.0 * random($seed, $x, 1);
    $t->[$x] = 0.25 + 9.75 * random($seed, $x, 2);
}

# CND(d) is written out twice: kernels have no subroutines.
sub price : GPU ($s, $k, $t, $call, $put, $parity, $r, $v) {
    my $sqrt_t = sqrt($t->[$x]);
    my $d1     = (log($s->[$x] / $k->[$x]) + ($r + 0.5 * $v * $v) * $t->[$x]) / ($v * $sqrt_t);
    my $d2     = $d1 - $v * $sqrt_t;
    my $k_n1   = 1.0 / (1.0 + 0.2316419 * abs($d1));
    my $n1     = 0.3989422804014327 * exp(-0.5 * $d1 * $d1) * (
        $k_n1 * (
            0.31938153 + $k_n1 * (-0.356563782 + $k_n1 * (1.781477937 + $k_n1 * (-1.821255978 + $k_n1 * 1.330274429)))
        )
    );
    $n1 = 1.0 - $n1 if $d1 > 0;
    my $k_n2 = 1.0 / (1.0 + 0.2316419 * abs($d2));
    my $n2   = 0.3989422804014327 * exp(-0.5 * $d2 * $d2) * (
        $k_n2 * (
            0.31938153 + $k_n2 * (-0.356563782 + $k_n2 * (1.781477937 + $k_n2 * (-1.821255978 + $k_n2 * 1.330274429)))
        )
    );
    $n2 = 1.0 - $n2 if $d2 > 0;
    my $discounted = $k->[$x] * exp(-$r * $t->[$x]);
    $call->[$x]   = $s->[$x] * $n1 - $discounted * $n2;
    $put->[$x]    = $discounted * (1.0 - $n2) - $s->[$x] * (1.0 - $n1);
    $parity->[$x] = abs($call->[$x] - $put->[$x] - ($s->[$x] - $discounted)) / $s->[$x];
}

configure(engine => $engine);

my ($s, $k, $t, $call, $put, $parity) = map { array(dims => [$n]) } 1 .. 6;
inputs($s, $k, $t, $seed);
price($s, $k, $t, $call, $put, $parity, $rate, $volatility);    # warm up
my $start = time;
price($s, $k, $t, $call, $put, $parity, $rate, $volatility);
my $worst_parity = $parity->max;    # waits for the kernel
my $elapsed = time - $start;

# reference: the exact normal distribution in double precision, first 1000 options
my sub cnd_exact ($d) { return 0.5 * erfc(-$d / sqrt(2)) }
my ($S, $K, $T, $C) = map { my @all = @{ $_->get }; [ @all[ 0 .. ($n < 1000 ? $n : 1000) - 1 ] ] } $s, $k, $t, $call;
my $worst = 0;
for my $i (0 .. $#$S) {
    my $d1    = (log($S->[$i] / $K->[$i]) + ($rate + $volatility**2 / 2) * $T->[$i]) / ($volatility * sqrt $T->[$i]);
    my $d2    = $d1 - $volatility * sqrt $T->[$i];
    my $exact = $S->[$i] * cnd_exact($d1) - $K->[$i] * exp(-$rate * $T->[$i]) * cnd_exact($d2);
    my $error = abs($C->[$i] - $exact);
    $worst = $error if $error > $worst;
}

printf "engine: %s\n%d options in %.3f ms (%.2f Goptions/s)\n", runner(\&price)->engine, $n, 1000 * $elapsed,
    $n / $elapsed / 1e9;
printf "max call price error vs exact normal CDF: %.2e; max put-call parity error: %.2e (relative)\n", $worst,
    $worst_parity;
say 'check: ', $worst < 1e-3 && $worst_parity < 1e-5 ? 'ok' : 'NG';

# Complementary error function (Numerical Recipes erfcc, relative error < 1.2e-7)
sub erfc ($z) {
    my $t = 1 / (1 + 0.5 * abs $z);
    my $r = $t * exp(
        -$z * $z - 1.26551223 + $t * (
            1.00002368 + $t * (
                0.37409196 + $t * (
                    0.09678418 + $t * (
                        -0.18628806 + $t * (
                            0.27886807 + $t * (-1.13520398 + $t * (1.48851587 + $t * (-0.82215223 + $t * 0.17087277)))
                        )
                    )
                )
            )
        )
    );
    return $z >= 0 ? $r : 2 - $r;
}
