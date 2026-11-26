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
use PerCUDA     qw(configure array runner);

my ($engine, $n, $tolerance) = @ARGV;
$n //= 512;    # interior points per side
$tolerance //= 1e-10;

# PI is visible to the kernels as a bareword.
use constant PI => 3.14159265358979;

my $h = 1 / ($n + 1);

# precision => 'double' (in configure) makes every array and scalar double.
sub init : GPU ($r, $p, $h) {
    my $px = ($x + 1) * $h;
    my $f  = (2.0 + PI * PI * $px * (1.0 - $px)) * sin(PI * ($y + 1) * $h);
    $r->[$x][$y] = $f;
    $p->[$x][$y] = $f;
}

# q = A p
sub apply : GPU ($p, $q, $inv_h2) {
    my $s = 4.0 * $p->[$x][$y];
    $s -= $p->[ $x - 1 ][$y] if $x > 0;
    $s -= $p->[ $x + 1 ][$y] if $x < $size_x - 1;
    $s -= $p->[$x][ $y - 1 ] if $y > 0;
    $s -= $p->[$x][ $y + 1 ] if $y < $size_y - 1;
    $q->[$x][$y] = $s * $inv_h2;
}

sub product : GPU ($a, $b, $w) { $w->[$x][$y] = $a->[$x][$y] * $b->[$x][$y] }

sub update : GPU ($u, $r, $p, $q, $alpha) {
    $u->[$x][$y] += $alpha * $p->[$x][$y];
    $r->[$x][$y] -= $alpha * $q->[$x][$y];
}

sub direction : GPU ($p, $r, $beta) { $p->[$x][$y] = $r->[$x][$y] + $beta * $p->[$x][$y] }

sub error : GPU ($u, $e, $h) {
    my $px = ($x + 1) * $h;
    $e->[$x][$y] = abs($u->[$x][$y] - $px * (1.0 - $px) * sin(PI * ($y + 1) * $h));
}

configure(engine => $engine, precision => 'double');

my $start = time;
my ($u, $r, $p, $q, $w) = map { array(dims => [ $n, $n ]) } 1 .. 5;
my sub dot ($a, $b) { product(a => $a, b => $b, w => $w); return $w->sum }

init(r => $r, p => $p, h => $h);
my $rr         = dot($r, $r);
my $rr0        = $rr;
my $iterations = 0;
while (sqrt($rr / $rr0) > $tolerance && $iterations < 10 * $n) {
    apply(p => $p, q => $q, inv_h2 => 1 / ($h * $h));
    my $alpha = $rr / dot($p, $q);
    update(u => $u, r => $r, p => $p, q => $q, alpha => $alpha);
    my $rr_new = dot($r, $r);
    direction(p => $p, r => $r, beta => $rr_new / $rr);
    $rr = $rr_new;
    $iterations++;
}
error(u => $u, e => $w, h => $h);
my $max_error = $w->max;
my $elapsed   = time - $start;

printf "engine: %s\n%d x %d unknowns, %d CG iterations, relative residual %.2e\n", runner(\&apply)->engine,
    $n, $n, $iterations,
    sqrt($rr / $rr0);
printf "max error: %.3e (truncation error O(h^2) ~ pi^2 h^2 / 48 = %.3e)\n", $max_error, PI**2 * $h**2 / 48;
printf "time: %.3f s\n", $elapsed;
say 'check: ', $max_error < PI**2 * $h**2 / 24 ? 'ok' : 'NG';
