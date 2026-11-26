use v5.40;
use Test2::V0;
use PerCUDA;

my sub runner ($source, %args) {
    return PerCUDA->new(
        engine   => 'perl',
        argument => [ in => 'i', out => 'f' ],
        retval   => ['out'],
        source   => $source,
        %args,
    );
}

my $in = [
    map {
        my $i = $_;
        [ map { $i * 7 - $_ * 3 } 0 .. 8 ]
    } 0 .. 12
];    # 13 x 9
my $kernel =
    'my ($x, $y) = thread_index(); my $s = 0.0; for my $k (0 .. $in->[$x][$y] % 5) { $s += $k / 3; } $out->[$x][$y] = $s;';
my $expected = runner($kernel)->run($in, []);

for my $jobs (2, 4, 13, 50, 'auto') {
    is scalar runner($kernel, perl_jobs => $jobs)->run($in, []), $expected, "perl_jobs => $jobs gives the same result";
}

# Stores into rows of other threads cannot be split; the kernel runs in one process.
my $other_rows = 'my ($x, $y) = thread_index(); $out->[12 - $x][$y] = $in->[$x][$y];';
is scalar runner($other_rows, perl_jobs => 4)->run($in, []), runner($other_rows)->run($in, []),
    'kernels writing other rows fall back to one process';

my $out = [ map { [ (5) x 9 ] } 0 .. 12 ];
runner('my ($x, $y) = thread_index(); $out->[$x][$y] = 1 if $y == 0;', perl_jobs => 3)->run($in, $out);
is [ map { "@$_" } @$out ], [ ("1 5 5 5 5 5 5 5 5") x 13 ], 'unwritten elements keep their values';

like dies { runner($kernel, perl_jobs => 0) },      qr/perl_jobs must be/, 'rejects 0';
like dies { runner($kernel, perl_jobs => 'many') }, qr/perl_jobs must be/, 'rejects words';

done_testing;
