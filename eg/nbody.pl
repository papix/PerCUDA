# Gravitational N-body simulation: all-pairs forces with Plummer softening
# (G = 1, total mass 1, eps = 0.05), integrated with the leapfrog
# (kick-drift-kick) scheme. Bodies start at rest, uniformly inside the unit
# sphere (a cold collapse). Leapfrog is symplectic, so the total energy stays
# close to its initial value; the relative drift is the check.
#
#   perl eg/nbody.pl [gpu|perl] [bodies] [steps] [dt]
use v5.40;
use FindBin;
use lib "$FindBin::Bin/../lib";
use Time::HiRes qw(time);
use PerCUDA     qw(configure array runner);

my ($engine, $n, $steps, $dt) = @ARGV;
$n     //= 8192;
$steps //= 1000;
$dt    //= 0.001;
my $eps2 = 0.05**2;

# Every kernel takes the masses m first, whether it uses them or not.
sub init : GPU ($m, $px, $py, $pz, $seed) {
    my $x   = thread_index();
    my $r   = random($seed, $x, 0)**(1.0 / 3.0);
    my $c   = 2.0 * random($seed, $x, 1) - 1.0;
    my $s   = sqrt(1.0 - $c * $c);
    my $phi = 6.283185307179586 * random($seed, $x, 2);
    $px->[$x] = $r * $s * cos($phi);
    $py->[$x] = $r * $s * sin($phi);
    $pz->[$x] = $r * $c;
}

sub accel : GPU ($m, $px, $py, $pz, $ax, $ay, $az, $eps2) {
    my $x      = thread_index();
    my $bodies = grid_size();
    my $sx     = 0.0;
    my $sy     = 0.0;
    my $sz     = 0.0;
    for my $j (0 .. $bodies - 1) {
        my $dx = $px->[$j] - $px->[$x];
        my $dy = $py->[$j] - $py->[$x];
        my $dz = $pz->[$j] - $pz->[$x];
        my $r2 = $dx * $dx + $dy * $dy + $dz * $dz + $eps2;
        my $w  = $m->[$j] / ($r2 * sqrt($r2));
        $sx += $w * $dx;
        $sy += $w * $dy;
        $sz += $w * $dz;
    }
    $ax->[$x] = $sx;
    $ay->[$x] = $sy;
    $az->[$x] = $sz;
}

sub kick : GPU ($m, $vx, $vy, $vz, $ax, $ay, $az, $h) {
    my $x = thread_index();
    $vx->[$x] += $h * $ax->[$x];
    $vy->[$x] += $h * $ay->[$x];
    $vz->[$x] += $h * $az->[$x];
}

sub drift : GPU ($m, $px, $py, $pz, $vx, $vy, $vz, $h) {
    my $x = thread_index();
    $px->[$x] += $h * $vx->[$x];
    $py->[$x] += $h * $vy->[$x];
    $pz->[$x] += $h * $vz->[$x];
}

# kinetic + half the pair potentials
sub energy : GPU ($m, $px, $py, $pz, $vx, $vy, $vz, $e, $eps2) {
    my $x      = thread_index();
    my $bodies = grid_size();
    my $pot    = 0.0;
    for my $j (0 .. $bodies - 1) {
        if ($j != $x) {
            my $dx = $px->[$j] - $px->[$x];
            my $dy = $py->[$j] - $py->[$x];
            my $dz = $pz->[$j] - $pz->[$x];
            $pot -= $m->[$j] / sqrt($dx * $dx + $dy * $dy + $dz * $dz + $eps2);
        }
    }
    my $v2 = $vx->[$x] * $vx->[$x] + $vy->[$x] * $vy->[$x] + $vz->[$x] * $vz->[$x];
    $e->[$x] = $m->[$x] * (0.5 * $v2 + 0.5 * $pot);
}

sub kinetic : GPU ($m, $vx, $vy, $vz, $e) {
    my $x = thread_index();
    $e->[$x] = 0.5 * $m->[$x] * ($vx->[$x] * $vx->[$x] + $vy->[$x] * $vy->[$x] + $vz->[$x] * $vz->[$x]);
}

configure(engine => $engine);

my $start = time;
my $m     = array([ (1 / $n) x $n ]);
my ($px, $py, $pz, $vx, $vy, $vz, $ax, $ay, $az, $e) = map { array(dims => [$n]) } 1 .. 10;
init($m, $px, $py, $pz, 1);
my sub total_energy () { energy($m, $px, $py, $pz, $vx, $vy, $vz, $e, $eps2); return $e->sum }

my $e0 = total_energy();
accel($m, $px, $py, $pz, $ax, $ay, $az, $eps2);
for (1 .. $steps) {
    kick($m, $vx, $vy, $vz, $ax, $ay, $az, $dt / 2);
    drift($m, $px, $py, $pz, $vx, $vy, $vz, $dt);
    accel($m, $px, $py, $pz, $ax, $ay, $az, $eps2);
    kick($m, $vx, $vy, $vz, $ax, $ay, $az, $dt / 2);
}
kinetic($m, $vx, $vy, $vz, $e);
my $k1      = $e->sum;
my $e1      = total_energy();
my $drift_e = abs(($e1 - $e0) / $e0);
my $elapsed = time - $start;

printf "engine: %s\n%d bodies, %d steps of dt = %g (%.2e pair interactions)\n", runner(\&accel)->engine, $n,
    $steps, $dt,
    $n * $n * $steps;
printf "energy: %.7f -> %.7f (kinetic %.7f), relative drift %.2e\n", $e0, $e1, $k1, $drift_e;
printf "time: %.3f s\n", $elapsed;
say 'check: ', $drift_e < 1e-3 ? 'ok' : 'NG';
