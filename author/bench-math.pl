# Benchmarks a kernel that is mostly math functions in every mode.
#
#   perl author/bench-math.pl [--size 1048576] [--perl-size 65536] [--runs 20]
#
# The gpu engine uses device arrays, so the time is the kernel's; the perl
# engine runs on fewer elements and is scaled to the same size.
use v5.40;
use FindBin;
use lib "$FindBin::Bin/../lib";
use Getopt::Long qw(GetOptions);
use Time::HiRes  qw(time);
use PerCUDA;

GetOptions(
    'size=i'      => \(my $size      = 1 << 20),
    'perl-size=i' => \(my $perl_size = 1 << 14),
    'runs=i'      => \(my $runs      = 20),
) or die "usage: $0 [--size N] [--perl-size N] [--runs N]\n";

my $KERNEL = <<'KERNEL';
my $x = thread_index();
my $v = $in->[$x];
my $s = 0.0;
for my $k (1 .. 8) {
    $s += exp(-$v * $k / 8) * sin($v * $k) + log(1.0 + $v * $k) * cos($v + $k) + atan2($v, $k) + $v ** 0.75;
}
$out->[$x] = $s;
KERNEL

my $gpu = eval { require CUDA::DriverAPI; CUDA::DriverAPI->new };
printf "%-7s %-7s %-6s %12s\n", 'engine', 'mode', 'type', 'ms / 1M elements';
for my $engine ($gpu ? qw(gpu perl) : qw(perl)) {
    for my $precision (qw(single double)) {
        my $type = $precision eq 'single' ? 'f' : 'd';
        for my $mode (qw(fast compat exact)) {
            my $n = $engine eq 'gpu' ? $size : $perl_size;
            my $r = PerCUDA->new(
                engine    => $engine,
                mode      => $mode,
                precision => $precision,
                perl_jobs => 'auto',
                argument  => [ in => $type, out => $type ],
                returns   => ['out'],
                source    => $KERNEL,
            );
            my $in  = $r->array([ map { $_ / $n * 10 } 1 .. $n ], type => $type);
            my $out = $r->array(type => $type, dims => [$n]);
            $r->run($in, $out);
            $out->get(into => my $first = []);    # warm up and wait
            my $count = $engine eq 'gpu' ? $runs : 1;
            my $start = time;
            $r->run($in, $out) for 1 .. $count;
            $out->get(into => my $result = []);
            my $ms = (time - $start) / $count * 1000 * ((1 << 20) / $n);
            printf "%-7s %-7s %-6s %12.3f\n", $engine, $mode, $precision, $ms;
        }
    }
}
