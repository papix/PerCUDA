# Benchmarks Conway's Game of Life, where each step reads the previous board.
#
#   perl author/bench-life.pl [--size 1024] [--steps 100] [--perl-jobs auto] [MODE ...]
#
# Modes (default: all available):
#   perl     perl engine, nested arrays
#   nested   gpu engine, nested arrays passed to every run
#   ndarray  gpu engine, PDL ndarrays passed to every run
#   device   gpu engine, boards kept on the device ($runner->array)
#
# Every mode starts from the same board and must end with the same one; the
# script prints the time per step and a checksum of the final board.
use v5.40;
use FindBin;
use lib "$FindBin::Bin/../lib";
use Getopt::Long qw(GetOptions);
use Time::HiRes  qw(time);
use PerCUDA;

# PDL is optional; import it at compile time so that long() and friends exist.
BEGIN {
    $main::HAVE_PDL = eval { require PDL; PDL->import; 1 }
}

GetOptions(
    'size=i'      => \(my $size      = 1024),
    'steps=i'     => \(my $steps     = 100),
    'perl-jobs=s' => \(my $perl_jobs = 'auto'),
) or die "usage: $0 [--size N] [--steps N] [--perl-jobs N|auto] [MODE ...]\n";

my $KERNEL = <<'KERNEL';
my $n = 0;
for my $dx (-1 .. 1) {
    for my $dy (-1 .. 1) {
        next if $dx == 0 && $dy == 0;
        $n += $cur->[($x + $dx + $size_x) % $size_x][($y + $dy + $size_y) % $size_y];
    }
}
$next->[$x][$y] = $n == 3 || ($n == 2 && $cur->[$x][$y]) ? 1 : 0;
KERNEL

sub runner ($engine) {
    return PerCUDA->new(
        dim       => 2,
        engine    => $engine,
        perl_jobs => $perl_jobs,
        size      => 'cur',
        argument  => [ cur => 'int', next => 'int' ],
        retval    => ['next'],
        source    => $KERNEL,
    );
}

# A reproducible random board (linear congruential generator).
my $seed  = 42;
my $board = [
    map {
        [ map { $seed = ($seed * 1103515245 + 12345) % 2**31; $seed >> 16 & 1 } 1 .. $size ]
    } 1 .. $size
];

sub checksum ($nested) {
    my $sum = 0;
    my $i   = 0;
    $sum = ($sum * 31 + $_ + $i++) % 1_000_000_007 for map { @$_ } @$nested;
    return $sum;
}

my $gpu   = eval { require CUDA::DriverAPI; CUDA::DriverAPI->new };
my $pdl   = $main::HAVE_PDL;
my %modes = (
    perl => sub {
        my $r = runner('perl');
        my ($cur, $next) = ([ map { [@$_] } @$board ], []);
        my $start = time;
        for (1 .. $steps) { $r->run($cur, $next); ($cur, $next) = ($next, $cur) }
        return (time - $start, $cur);
    },
    nested => sub {
        my $r = runner('gpu');
        my ($cur, $next) = ([ map { [@$_] } @$board ], []);
        $r->run($cur, []);    # warm up (first launch)
        my $start = time;
        for (1 .. $steps) { $r->run($cur, $next); ($cur, $next) = ($next, $cur) }
        return (time - $start, $cur);
    },
    ndarray => sub {
        my $r = runner('gpu');
        my ($cur, $next) = (pdl(long(), $board), zeroes(long(), $size, $size));
        $r->run($cur, zeroes(long(), $size, $size));
        my $start = time;
        for (1 .. $steps) { $r->run($cur, $next); ($cur, $next) = ($next, $cur) }
        my $time = time - $start;
        return ($time, $cur->unpdl);
    },
    device => sub {
        my $r = runner('gpu');
        my ($cur, $next) = ($r->array($board, type => 'int'), $r->array(type => 'int', dims => [ $size, $size ]));
        $r->run($cur, $next);
        $cur = $r->array($board, type => 'int');
        my $start = time;
        for (1 .. $steps) { $r->run($cur, $next); ($cur, $next) = ($next, $cur) }
        my $result = $cur->get;    # the final board comes back once
        return (time - $start, $result);
    },
);
my %available = (
    perl    => 1,
    nested  => $gpu,
    ndarray => $gpu && $pdl,
    device  => $gpu && PerCUDA->can('array'),
);

# perl runs last: its large allocations slow down the modes measured after it.
my @modes = @ARGV ? @ARGV : grep { $available{$_} } qw(nested ndarray device perl);
printf "Game of Life, %d x %d, %d steps\n", $size, $size, $steps;
for my $mode (@modes) {
    die "unknown mode: $mode\n" unless $modes{$mode};
    if (!$available{$mode}) {
        printf "%-8s unavailable\n", $mode;
        next;
    }
    my ($time, $final) = $modes{$mode}->();
    printf "%-8s %9.3f ms/step  %8.3f s total  checksum %d\n", $mode, 1000 * $time / $steps, $time, checksum($final);
}
