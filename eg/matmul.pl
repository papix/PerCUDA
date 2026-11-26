use v5.40;
use FindBin;
use lib "$FindBin::Bin/../lib";
use PerCUDA qw(configure runner);

my $array = [ [ 1 .. 3 ], [ 4 .. 6 ], [ 7 .. 9 ], ];

# The kernel is a Perl sub with the :GPU attribute; its signature orders the
# arguments, and calling it runs it.
sub matmul : GPU ($array1, $array2, $array3, $N) {
    my $tmp = 0;
    for (my $i = 0; $i < $N; $i++) {
        $tmp = $tmp + $array1->[$i]->[$y] * $array2->[$x]->[$i];
    }
    $array3->[$x]->[$y] = $tmp;
}

# The types (float arrays; N is an int, as a loop bound), dim (2, from $y),
# size (array1) and retval (array3, the array written) are inferred.
configure(engine => $ARGV[0]);

my $retval = matmul($array, $array, [], 3);
say 'engine: ', runner(\&matmul)->engine;
say join ' ',   @$_ for @$retval;
