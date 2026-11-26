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
use PerCUDA     qw(configure array runner);

my ($engine, $n, $steps) = @ARGV;
$n     //= 256;
$steps //= 2000;
my $alpha = 1.0;
my $h     = 1 / ($n - 1);
my $r     = 0.24;    # alpha dt / h^2; FTCS is stable for r <= 1/4
my $dt    = $r * $h * $h / $alpha;

# Constants are visible to the kernels as barewords.
use constant PI => 3.14159265358979;

# precision => 'double' (in configure) makes every array and scalar double.
sub init : GPU ($u, $h) {
    my ($x, $y) = thread_index();
    $u->[$x][$y] = sin(PI * $x * $h) * sin(PI * $y * $h);
}

sub step : GPU ($u, $v, $r) {
    my ($x,      $y)      = thread_index();
    my ($size_x, $size_y) = grid_size();
    if ($x == 0 || $y == 0 || $x == $size_x - 1 || $y == $size_y - 1) {
        $v->[$x][$y] = 0.0;
    }
    else {
        $v->[$x][$y] =
            $u->[$x][$y] +
            $r *
            ($u->[ $x - 1 ][$y] + $u->[ $x + 1 ][$y] + $u->[$x][ $y - 1 ] + $u->[$x][ $y + 1 ] - 4.0 * $u->[$x][$y]);
    }
}

sub error : GPU ($u, $e, $decay, $h) {
    my ($x, $y) = thread_index();
    $e->[$x][$y] = abs($u->[$x][$y] - $decay * sin(PI * $x * $h) * sin(PI * $y * $h));
}

configure(engine => $engine, precision => 'double');

my $start = time;
my ($u, $v) = map { array(dims => [ $n, $n ]) } 1 .. 2;
init($u, $h);
for (1 .. $steps) {
    step($u, $v, $r);
    ($u, $v) = ($v, $u);
}
my $decay = exp(-2 * PI * PI * $alpha * $steps * $dt);
my $e     = array(dims => [ $n, $n ]);
error($u, $e, $decay, $h);
my $relative = $e->max / $decay;
my $elapsed  = time - $start;

printf "engine: %s\n%d x %d grid, %d steps, t = %.3e\n", runner(\&step)->engine, $n, $n, $steps, $steps * $dt;
printf "max error relative to the peak: %.3e (truncation error O(h^2) ~ %.1e)\n", $relative, PI**2 * $h**2 / 12;
printf "time: %.3f s\n",                                                          $elapsed;
say 'check: ', $relative < PI**2 * $h**2 ? 'ok' : 'NG';
