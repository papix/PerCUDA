# Poisson equation -(u_xx + u_yy) = f on the unit square with u = 0 on the
# boundary, discretized with the 5-point stencil and solved with the
# conjugate gradient method, matrix free. f = (2 + pi^2 x (1 - x)) sin(pi y)
# has the solution u = x (1 - x) sin(pi y); the discretization error is O(h^2).
# Every vector stays on the device; dot products are reductions.
#
#   perl eg/poisson-cg.pl [gpu|perl] [n] [tolerance]
use v5.40;
use FindBin;
use lib "$FindBin::Bin/../lib";
use Time::HiRes qw(time);
use PerCUDA;

my ($engine, $n, $tolerance) = @ARGV;
$n //= 512;    # interior points per side
$tolerance //= 1e-10;
my $h  = 1 / ($n + 1);
my $PI = 3.14159265358979;

my %common = (engine => $engine, dim => 2, precision => 'double');
my sub kernel ($argument, $source) { return PerCUDA->new(%common, argument => $argument, source => $source) }

my $init = kernel([ r => 'd', p => 'd', h => 'd' ], <<"KERNEL");
my \$px = (\$x + 1) * \$h;
my \$f = (2.0 + $PI * $PI * \$px * (1.0 - \$px)) * sin($PI * (\$y + 1) * \$h);
\$r->[\$x][\$y] = \$f;
\$p->[\$x][\$y] = \$f;
KERNEL
my $apply = kernel([ p => 'd', q => 'd', inv_h2 => 'd' ], <<'KERNEL');    # q = A p
my $s = 4.0 * $p->[$x][$y];
$s -= $p->[$x - 1][$y] if $x > 0;
$s -= $p->[$x + 1][$y] if $x < $size_x - 1;
$s -= $p->[$x][$y - 1] if $y > 0;
$s -= $p->[$x][$y + 1] if $y < $size_y - 1;
$q->[$x][$y] = $s * $inv_h2;
KERNEL
my $product = kernel([ a => 'd', b => 'd', w => 'd' ], '$w->[$x][$y] = $a->[$x][$y] * $b->[$x][$y];');
my $update  = kernel([ u => 'd', r => 'd', p => 'd', q => 'd', alpha => 'd' ], <<'KERNEL');
$u->[$x][$y] += $alpha * $p->[$x][$y];
$r->[$x][$y] -= $alpha * $q->[$x][$y];
KERNEL
my $direction = kernel([ p => 'd', r => 'd', beta => 'd' ], '$p->[$x][$y] = $r->[$x][$y] + $beta * $p->[$x][$y];');
my $error     = kernel([ u => 'd', e => 'd', h    => 'd' ], <<"KERNEL");
my \$px = (\$x + 1) * \$h;
\$e->[\$x][\$y] = abs(\$u->[\$x][\$y] - \$px * (1.0 - \$px) * sin($PI * (\$y + 1) * \$h));
KERNEL

my $start = time;
my ($u, $r, $p, $q, $w) = map { $apply->array(type => 'd', dims => [ $n, $n ]) } 1 .. 5;
my sub dot ($a, $b) { $product->run($a, $b, $w); return $w->sum }

$init->run($r, $p, $h);
my $rr         = dot($r, $r);
my $rr0        = $rr;
my $iterations = 0;
while (sqrt($rr / $rr0) > $tolerance && $iterations < 10 * $n) {
    $apply->run($p, $q, 1 / ($h * $h));
    my $alpha = $rr / dot($p, $q);
    $update->run($u, $r, $p, $q, $alpha);
    my $rr_new = dot($r, $r);
    $direction->run($p, $r, $rr_new / $rr);
    $rr = $rr_new;
    $iterations++;
}
$error->run($u, $w, $h);
my $max_error = $w->max;
my $elapsed   = time - $start;

printf "engine: %s\n%d x %d unknowns, %d CG iterations, relative residual %.2e\n", $apply->engine, $n, $n, $iterations,
    sqrt($rr / $rr0);
printf "max error: %.3e (truncation error O(h^2) ~ pi^2 h^2 / 48 = %.3e)\n", $max_error, $PI**2 * $h**2 / 48;
printf "time: %.3f s\n", $elapsed;
say 'check: ', $max_error < $PI**2 * $h**2 / 24 ? 'ok' : 'NG';
