use v5.40;
use Test2::V0;
use lib 't/lib';
use PerCUDATest;
use PerCUDA;

# The example of Fig. 2 in the paper, with N passed explicitly.
my $KERNEL = <<'K';
my ($x, $y) = thread_index();
my $tmp = 0;
for (my $i = 0; $i < $N; $i++) {
    $tmp = $tmp
        + $array1->[$i]->[$y]
        * $array2->[$x]->[$i];
}
$array3->[$x]->[$y] = $tmp;
K

sub runner ($engine) {
    return PerCUDA->new(
        size     => 'array1',
        engine   => $engine,
        argument => [ array1 => 'f', array2 => 'f', array3 => 'f', N => 'i' ],
        returns  => ['array3'],
        source   => $KERNEL,
    );
}

sub reference ($a, $b, $n) {
    my @c;
    for my $x (0 .. $n - 1) {
        for my $y (0 .. $n - 1) {
            my $s = 0;
            $s += $a->[$_][$y] * $b->[$x][$_] for 0 .. $n - 1;
            $c[$x][$y] = $s;
        }
    }
    return \@c;
}

my $array    = [ [ 1 .. 3 ], [ 4 .. 6 ], [ 7 .. 9 ] ];
my $expected = [ [ 30, 36, 42 ], [ 66, 81, 96 ], [ 102, 126, 150 ] ];

for my $engine (both_engines()) {
    subtest $engine => sub {
        my $r = runner($engine);
        is $r->engine, $engine, 'engine';
        my $out     = [];
        my $returns = $r->run($array, $array, $out, 3);
        is $returns, $expected, '3x3 product';
        is $out,     $expected, 'returns array is updated in place';

        my $n = 64;
        my $a = [
            map {
                my $i = $_;
                [ map { ($i * 7 + $_ * 3) % 11 - 5 } 0 .. $n - 1 ]
            } 0 .. $n - 1
        ];
        my $b = [
            map {
                my $i = $_;
                [ map { ($i * 5 + $_) % 13 - 6 } 0 .. $n - 1 ]
            } 0 .. $n - 1
        ];
        is scalar $r->run($a, $b, [], $n), reference($a, $b, $n), '64x64 product (non-multiple of block size grid)';
    };
}

like dies { runner('perl')->run($array, $array, []) }, qr/run expects 4 arguments but got 3/,
    'argument count is checked';

is runner(undef)->engine, gpu_available() ? 'gpu' : 'perl', 'engine is auto-detected';

done_testing;
