use v5.40;
use Test2::V0;
use File::Temp qw(tempdir);
use FindBin;

# Run the examples in eg/ with small inputs on every available engine, so that
# the documentation stays working code.
my $eg = "$FindBin::Bin/../eg";
local $ENV{PERCUDA_CACHE_DIR} = tempdir(CLEANUP => 1);

my @engines = ('perl', (eval { require CUDA::DriverAPI; CUDA::DriverAPI->new } ? 'gpu' : ()));

sub run_example (@command) {
    my $output = qx{$^X @command 2>/dev/null};
    is $?, 0, "@command exits successfully";
    return $output;
}

for my $engine (@engines) {
    like run_example("$eg/matmul.pl", $engine), qr/engine: $engine\n30 36 42\n66 81 96\n102 126 150\n/,
        "matmul.pl ($engine)";
    like run_example("$eg/mandelbrot.pl", $engine, 32,    16), qr/\AP2\n32 32\n255\n/, "mandelbrot.pl ($engine)";
    like run_example("$eg/pi.pl",         $engine, 20000, 1),  qr/engine: $engine\npi = 3\.1/, "pi.pl ($engine)";
    like run_example("$eg/benchmark.pl",  $engine, 16), qr/engine=$engine n=16 round=1: .* \(c\[1\]\[2\]=\d+\)/,
        "benchmark.pl ($engine)";
}

done_testing;
