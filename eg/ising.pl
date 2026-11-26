# 2D Ising model (J = 1, no field, periodic L x L lattice) sampled with the
# Metropolis algorithm on a checkerboard: the two colours are updated in
# turn, so every site of one colour can flip in parallel. The acceptance
# probabilities exp(-4/T) and exp(-8/T) are computed once on the host, as
# usual. The mean |magnetization| per site is compared with Onsager's exact
# result m = (1 - sinh(2/T)^-4)^(1/8) below Tc = 2/ln(1 + sqrt 2) = 2.269,
# and with ~0 well above it.
#
#   perl eg/ising.pl [gpu|perl] [L] [sweeps] [T ...]
use v5.40;
use FindBin;
use lib "$FindBin::Bin/../lib";
use Time::HiRes qw(time);
use PerCUDA     qw(configure array runner);

my ($engine, $L, $sweeps, @temperatures) = @ARGV;
$L      //= 256;
$sweeps //= 2000;    # half for equilibration, half for measurement
@temperatures = (1.5, 2.0, 2.269, 2.5, 3.5) unless @temperatures;

# The spins are an int array (+1 or -1); p4 and p8 are float acceptance probabilities.
sub metropolis : GPU({ argument => [ spin => 'int', parity => 'int' ] }) ($spin, $parity, $seed, $step, $p4, $p8) {
    my ($x,      $y)      = thread_index();
    my ($size_x, $size_y) = grid_size();
    if (($x + $y) % 2 == $parity) {
        my $s = $spin->[$x][$y];
        my $neighbours =
            $spin->[ ($x + 1) % $size_x ][$y] +
            $spin->[ ($x + $size_x - 1) % $size_x ][$y] +
            $spin->[$x][ ($y + 1) % $size_y ] +
            $spin->[$x][ ($y + $size_y - 1) % $size_y ];
        my $de = 2 * $s * $neighbours;
        my $p  = $de <= 0 ? 1.0 : $de == 4 ? $p4 : $p8;
        $spin->[$x][$y] = -$s if random($seed, $x, $y, $step) < $p;
    }
}

configure(engine => $engine);

my $start = time;
my $ok    = 1;
printf "engine: %s\n%d x %d lattice, %d sweeps per temperature\n", runner(\&metropolis)->engine, $L, $L, $sweeps;
printf "%6s %10s %10s\n", 'T', '<|m|>', 'Onsager';
for my $i (0 .. $#temperatures) {
    my $t    = $temperatures[$i];
    my $spin = array([ map { [ (1) x $L ] } 1 .. $L ], type => 'int');    # ordered start
    my ($p4, $p8)     = (exp(-4 / $t), exp(-8 / $t));
    my ($sum, $count) = (0, 0);
    for my $s (1 .. $sweeps) {
        metropolis($spin, $_, $i + 1, $s, $p4, $p8) for 0, 1;
        next if $s <= $sweeps / 2;
        $sum += abs($spin->sum) / ($L * $L);
        $count++;
    }
    my $m     = $sum / $count;
    my $exact = $t < 2.269185 ? (1 - sinh(2 / $t)**-4)**(1 / 8) : 0;
    printf "%6.3f %10.4f %10s\n", $t, $m, $t < 2.269185 ? sprintf('%.4f', $exact) : '0';

    # far from Tc the finite lattice agrees with the infinite one
    $ok &&= abs($m - $exact) < 0.02 if $t <= 2.0;
    $ok &&= $m < 0.2                if $t >= 3.0;
}
printf "time: %.3f s\n", time - $start;
say 'check: ', $ok ? 'ok' : 'NG';

sub sinh ($v) { return (exp($v) - exp(-$v)) / 2 }
