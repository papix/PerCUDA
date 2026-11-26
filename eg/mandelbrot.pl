# Usage: perl examples/mandelbrot.pl [gpu|perl] [width] [max_iterations] > mandelbrot.pgm
use v5.40;
use FindBin;
use lib "$FindBin::Bin/../lib";
use Time::HiRes qw(time);
use PerCUDA     qw(configure runner);

my ($engine, $width, $max) = @ARGV;
$width //= 1024;
$max   //= 256;

# The image holds iteration counts, so it is an int array.
sub mandelbrot : GPU({ argument => [ image => 'int', max => 'int' ] }) ($image, $max) {
    my $cr = -2.0 + 3.0 * $y / $size_y;
    my $ci = -1.5 + 3.0 * $x / $size_x;
    my $zr = 0.0;
    my $zi = 0.0;
    my $n  = 0;
    while ($n < $max && $zr * $zr + $zi * $zi <= 4.0) {
        my $t = $zr * $zr - $zi * $zi + $cr;
        $zi = 2.0 * $zr * $zi + $ci;
        $zr = $t;
        $n++;
    }
    $image->[$x][$y] = $n;
}

configure(engine => $engine);

my $image = [ map { [ (0) x $width ] } 1 .. $width ];
my $start = time;
mandelbrot($image, $max);
printf STDERR "engine=%s size=%dx%d max=%d: %.5fs\n", runner(\&mandelbrot)->engine, $width, $width, $max, time - $start;

print "P2\n$width $width\n255\n";
say join ' ', map { $_ >= $max ? 0 : 255 - int(255 * $_ / $max) } @$_ for @$image;
