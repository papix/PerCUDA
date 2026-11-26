# Gray-Scott reaction-diffusion system
#
#   u_t = Du lap(u) - u v^2 + F (1 - u)
#   v_t = Dv lap(v) + u v^2 - (F + k) v
#
# on a periodic grid with the parameters popularized by Karl Sims' tutorial
# (Du = 1.0, Dv = 0.5, F = 0.055, k = 0.062, dt = 1, 3 x 3 Laplacian with
# weights -1 / 0.2 / 0.05), which grow "coral" patterns from a seeded square.
# The final v is written as a PGM image if a file name is given.
#
#   perl eg/gray-scott.pl [gpu|perl] [n] [steps] [image.pgm] [F] [k]
use v5.40;
use FindBin;
use lib "$FindBin::Bin/../lib";
use Time::HiRes qw(time);
use PerCUDA     qw(configure array runner);

my ($engine, $n, $steps, $image, $feed, $kill) = @ARGV;
$n     //= 256;
$steps //= 10000;
$feed  //= 0.055;
$kill  //= 0.062;
my ($du, $dv, $dt) = (1.0, 0.5, 1.0);

sub init : GPU ($u, $v, $seed) {
    my ($x,      $y)      = thread_index();
    my ($size_x, $size_y) = grid_size();
    my $inside = abs($x - $size_x / 2) < $size_x / 10 && abs($y - $size_y / 2) < $size_y / 10;
    $u->[$x][$y] = $inside ? 0.5                                : 1.0;
    $v->[$x][$y] = $inside ? 0.25 + 0.1 * random($seed, $x, $y) : 0.0;
}

sub step : GPU ($u, $v, $u2, $v2, $du, $dv, $feed, $kill, $dt) {
    my ($x, $y)           = thread_index();
    my ($size_x, $size_y) = grid_size();
    my $xm = ($x + $size_x - 1) % $size_x;
    my $xp = ($x + 1) % $size_x;
    my $ym = ($y + $size_y - 1) % $size_y;
    my $yp = ($y + 1) % $size_y;
    my $a  = $u->[$x][$y];
    my $b  = $v->[$x][$y];
    my $la =
        -$a +
        0.2 * ($u->[$xm][$y] + $u->[$xp][$y] + $u->[$x][$ym] + $u->[$x][$yp]) +
        0.05 * ($u->[$xm][$ym] + $u->[$xm][$yp] + $u->[$xp][$ym] + $u->[$xp][$yp]);
    my $lb =
        -$b +
        0.2 * ($v->[$xm][$y] + $v->[$xp][$y] + $v->[$x][$ym] + $v->[$x][$yp]) +
        0.05 * ($v->[$xm][$ym] + $v->[$xm][$yp] + $v->[$xp][$ym] + $v->[$xp][$yp]);
    my $abb = $a * $b * $b;
    $u2->[$x][$y] = $a + $dt * ($du * $la - $abb + $feed * (1.0 - $a));
    $v2->[$x][$y] = $b + $dt * ($dv * $lb + $abb - ($feed + $kill) * $b);
}

configure(engine => $engine);

my $start = time;
my ($u, $v, $u2, $v2) = map { array(dims => [ $n, $n ]) } 1 .. 4;
init($u, $v, 1);
for (1 .. $steps) {
    step($u, $v, $u2, $v2, $du, $dv, $feed, $kill, $dt);
    ($u, $u2, $v, $v2) = ($u2, $u, $v2, $v);
}
my ($min, $max, $mean) = ($v->min, $v->max, $v->sum / ($n * $n));
my $elapsed = time - $start;

printf "engine: %s\n%d x %d grid, %d steps, F = %g, k = %g\n", runner(\&step)->engine, $n, $n, $steps, $feed, $kill;
printf "v: min %.4f, max %.4f, mean %.4f\n", $min, $max, $mean;
printf "time: %.3f s\n",                     $elapsed;
if (defined $image) {
    open my $fh, '>', $image or die "$image: $!";
    print {$fh} "P2\n$n $n\n255\n";
    print {$fh} join(' ', map { my $c = int(255 * $_ / ($max || 1)); $c < 0 ? 0 : $c } @$_), "\n" for @{ $v->get };
    close $fh;
    say "wrote $image";
}
say 'check: ', $max > 0.1 && $max < 1 && $min >= 0 ? 'ok' : 'NG';    # the pattern grew and stayed bounded
