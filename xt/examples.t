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

    # research samples with sizes small enough for the perl engine
    for my $sample (
        [ 'heat.pl',          24, 50 ],
        [ 'poisson-cg.pl',    16 ],
        [ 'gray-scott.pl',    40, 300 ],
        [ 'nbody.pl',         64, 20 ],
        [ 'ising.pl',         16, 200, 1.5, 3.5 ],
        [ 'black-scholes.pl', 2000 ],
        [ 'mc-option.pl',     20000 ],
        [ 'kmeans.pl',        3000, 4 ],
        )
    {
        my ($script, @args) = @$sample;
        like run_example("$eg/$script", $engine, @args), qr/engine: $engine\n.*\ncheck: ok\n\z/s, "$script ($engine)";
    }
    like run_example("$eg/benchmark.pl", $engine, 16), qr/engine=$engine n=16 round=1: .* \(c\[1\]\[2\]=\d+\)/,
        "benchmark.pl ($engine)";
}

done_testing;
