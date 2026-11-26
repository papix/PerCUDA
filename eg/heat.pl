# 2D heat equation u_t = alpha (u_xx + u_yy) on the unit square with u = 0 on
# the boundary, solved with the explicit FTCS scheme (forward Euler in time,
# 5-point Laplacian in space). The initial condition sin(pi x) sin(pi y) has
# the exact solution exp(-2 pi^2 alpha t) sin(pi x) sin(pi y), which the
# result is checked against.
#
#   perl eg/heat.pl [gpu|perl] [n] [steps]
use v5.40;
use FindBin;
use lib "$FindBin::Bin/../lib";
use Time::HiRes qw(time);
use PerCUDA;

my ($engine, $n, $steps) = @ARGV;
$n     //= 256;
$steps //= 2000;
my $alpha = 1.0;
my $h     = 1 / ($n - 1);
my $r     = 0.24;    # alpha dt / h^2; FTCS is stable for r <= 1/4
my $dt    = $r * $h * $h / $alpha;
my $PI    = 3.14159265358979;

my %common = (engine => $engine, dim => 2, precision => 'double');
my $init   = PerCUDA->new(%common, argument => [ u => 'd', h => 'd' ], source => <<"KERNEL");
\$u->[\$x][\$y] = sin($PI * \$x * \$h) * sin($PI * \$y * \$h);
KERNEL
my $step = PerCUDA->new(%common, size => 'u', argument => [ u => 'd', v => 'd', r => 'd' ], source => <<'KERNEL');
if ($x == 0 || $y == 0 || $x == $size_x - 1 || $y == $size_y - 1) {
    $v->[$x][$y] = 0.0;
}
else {
    $v->[$x][$y] = $u->[$x][$y]
        + $r * ($u->[$x - 1][$y] + $u->[$x + 1][$y] + $u->[$x][$y - 1] + $u->[$x][$y + 1] - 4.0 * $u->[$x][$y]);
}
KERNEL
my $error = PerCUDA->new(
    %common,
    size     => 'u',
    argument => [ u => 'd', e => 'd', decay => 'd', h => 'd' ],
    source   => <<"KERNEL");
\$e->[\$x][\$y] = abs(\$u->[\$x][\$y] - \$decay * sin($PI * \$x * \$h) * sin($PI * \$y * \$h));
KERNEL

my $start = time;
my ($u, $v) = map { $step->array(type => 'd', dims => [ $n, $n ]) } 1 .. 2;
$init->run($u, $h);
for (1 .. $steps) {
    $step->run($u, $v, $r);
    ($u, $v) = ($v, $u);
}
my $decay = exp(-2 * $PI * $PI * $alpha * $steps * $dt);
my $e     = $step->array(type => 'd', dims => [ $n, $n ]);
$error->run($u, $e, $decay, $h);
my $relative = $e->max / $decay;
my $elapsed  = time - $start;

printf "engine: %s\n%d x %d grid, %d steps, t = %.3e\n", $step->engine, $n, $n, $steps, $steps * $dt;
printf "max error relative to the peak: %.3e (truncation error O(h^2) ~ %.1e)\n", $relative, $PI**2 * $h**2 / 12;
printf "time: %.3f s\n",                                                          $elapsed;
say 'check: ', $relative < $PI**2 * $h**2 ? 'ok' : 'NG';
