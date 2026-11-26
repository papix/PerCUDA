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
use PerCUDA;

my ($engine, $n, $steps, $dt) = @ARGV;
$n     //= 8192;
$steps //= 1000;
$dt    //= 0.001;
my $eps2 = 0.05**2;

my sub kernel ($argument, $source) {
    return PerCUDA->new(engine => $engine, size => 'm', argument => [ m => 'f', @$argument ], source => $source);
}
my @p = (px => 'f', py => 'f', pz => 'f');
my @v = (vx => 'f', vy => 'f', vz => 'f');
my @a = (ax => 'f', ay => 'f', az => 'f');

my $init = kernel([ @p, seed => 'i' ], <<'KERNEL');
my $r = random($seed, $x, 0) ** (1.0 / 3.0);
my $c = 2.0 * random($seed, $x, 1) - 1.0;
my $s = sqrt(1.0 - $c * $c);
my $phi = 6.283185307179586 * random($seed, $x, 2);
$px->[$x] = $r * $s * cos($phi);
$py->[$x] = $r * $s * sin($phi);
$pz->[$x] = $r * $c;
KERNEL
my $accel = kernel([ @p, @a, eps2 => 'f' ], <<'KERNEL');
my $sx = 0.0;
my $sy = 0.0;
my $sz = 0.0;
for my $j (0 .. $size_x - 1) {
    my $dx = $px->[$j] - $px->[$x];
    my $dy = $py->[$j] - $py->[$x];
    my $dz = $pz->[$j] - $pz->[$x];
    my $r2 = $dx * $dx + $dy * $dy + $dz * $dz + $eps2;
    my $w = $m->[$j] / ($r2 * sqrt($r2));
    $sx += $w * $dx;
    $sy += $w * $dy;
    $sz += $w * $dz;
}
$ax->[$x] = $sx;
$ay->[$x] = $sy;
$az->[$x] = $sz;
KERNEL
my $kick = kernel([ @v, @a, h => 'f' ], <<'KERNEL');
$vx->[$x] += $h * $ax->[$x];
$vy->[$x] += $h * $ay->[$x];
$vz->[$x] += $h * $az->[$x];
KERNEL
my $drift = kernel([ @p, @v, h => 'f' ], <<'KERNEL');
$px->[$x] += $h * $vx->[$x];
$py->[$x] += $h * $vy->[$x];
$pz->[$x] += $h * $vz->[$x];
KERNEL
my $energy = kernel([ @p, @v, e => 'f', eps2 => 'f' ], <<'KERNEL');    # kinetic + half the pair potentials
my $pot = 0.0;
for my $j (0 .. $size_x - 1) {
    if ($j != $x) {
        my $dx = $px->[$j] - $px->[$x];
        my $dy = $py->[$j] - $py->[$x];
        my $dz = $pz->[$j] - $pz->[$x];
        $pot -= $m->[$j] / sqrt($dx * $dx + $dy * $dy + $dz * $dz + $eps2);
    }
}
my $v2 = $vx->[$x] * $vx->[$x] + $vy->[$x] * $vy->[$x] + $vz->[$x] * $vz->[$x];
$e->[$x] = $m->[$x] * (0.5 * $v2 + 0.5 * $pot);
KERNEL

my $kinetic = kernel([ @v, e => 'f' ], <<'KERNEL');
$e->[$x] = 0.5 * $m->[$x] * ($vx->[$x] * $vx->[$x] + $vy->[$x] * $vy->[$x] + $vz->[$x] * $vz->[$x]);
KERNEL

my $start = time;
my $m     = $accel->array([ (1 / $n) x $n ], type => 'f');
my ($px, $py, $pz, $vx, $vy, $vz, $ax, $ay, $az, $e) = map { $accel->array(type => 'f', dims => [$n]) } 1 .. 10;
$init->run($m, $px, $py, $pz, 1);
my sub total_energy () { $energy->run($m, $px, $py, $pz, $vx, $vy, $vz, $e, $eps2); return $e->sum }

my $e0 = total_energy();
$accel->run($m, $px, $py, $pz, $ax, $ay, $az, $eps2);
for (1 .. $steps) {
    $kick->run($m, $vx, $vy, $vz, $ax, $ay, $az, $dt / 2);
    $drift->run($m, $px, $py, $pz, $vx, $vy, $vz, $dt);
    $accel->run($m, $px, $py, $pz, $ax, $ay, $az, $eps2);
    $kick->run($m, $vx, $vy, $vz, $ax, $ay, $az, $dt / 2);
}
$kinetic->run($m, $vx, $vy, $vz, $e);
my $k1      = $e->sum;
my $e1      = total_energy();
my $drift_e = abs(($e1 - $e0) / $e0);
my $elapsed = time - $start;

printf "engine: %s\n%d bodies, %d steps of dt = %g (%.2e pair interactions)\n", $accel->engine, $n, $steps, $dt,
    $n * $n * $steps;
printf "energy: %.7f -> %.7f (kinetic %.7f), relative drift %.2e\n", $e0, $e1, $k1, $drift_e;
printf "time: %.3f s\n", $elapsed;
say 'check: ', $drift_e < 1e-3 ? 'ok' : 'NG';
