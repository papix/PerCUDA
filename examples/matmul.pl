use v5.40;
use FindBin;
use lib "$FindBin::Bin/../lib";
use PerCUDA;

my $array = [
    [ 1 .. 3 ],
    [ 4 .. 6 ],
    [ 7 .. 9 ],
];

my $runner = PerCUDA->new(
    dim      => 2,
    size     => 'array1',
    engine   => $ARGV[0],
    argument => [
        array1 => 'f',
        array2 => 'f',
        array3 => 'f',
        N      => 'i',
    ],
    retval => ['array3'],
    source => <<'KERNEL',
my $tmp = 0;
for (my $i = 0; $i < $N; $i++) {
    $tmp = $tmp
        + $array1->[$i]->[$y]
        * $array2->[$x]->[$i];
}
$array3->[$x]->[$y] = $tmp;
KERNEL
);

my $retval = $runner->run($array, $array, [], 3);
say 'engine: ', $runner->engine;
say join ' ', @$_ for @$retval;
