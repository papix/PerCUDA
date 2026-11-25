use v5.40;
use Test2::V0 '!float';    # PDL exports float (the type)
use lib 't/lib';
use PerCUDATest;
use PerCUDA;

use Test2::Require::Module qw(PDL);
use PDL;

my %args = (
    dim      => 2,
    size     => 'c',
    argument => [ a => 'f', b => 'f', c => 'f', n => 'i' ],
    retval   => ['c'],
    source   => 'my $s = 0.0; for (my $k = 0; $k < $n; $k++) { $s += $a->[$x][$k] * $b->[$k][$y]; } $c->[$x][$y] = $s;',
);

my $nested_a = [
    map {
        my $i = $_;
        [ map { ($i * 3 + $_) % 7 - 3 } 0 .. 4 ]
    } 0 .. 3
];    # 4 x 5
my $nested_b = [
    map {
        my $i = $_;
        [ map { ($i + $_ * 2) % 5 - 2 } 0 .. 5 ]
    } 0 .. 4
];    # 5 x 6
my $nested_c = [ map { [ (0) x 6 ] } 1 .. 4 ];
my $expected = PerCUDA->new(%args, engine => 'perl')->run($nested_a, $nested_b, $nested_c, 5);    # 4 x 6

for my $engine (ndarray_engines()) {
    subtest $engine => sub {
        my $r = PerCUDA->new(%args, engine => $engine);
        my ($a, $b) = map { pdl(float, $_) } $nested_a, $nested_b;
        is [ $a->dims ], [ 5, 4 ], 'an ndarray lists the innermost dimension first';

        my $c      = zeroes(float, 6, 4);
        my $result = $r->run($a, $b, $c, 5);
        ref_is $result, $c, 'the ndarray passed for the result is returned';
        is [ $c->dims ], [ 6, 4 ],  'its dimensions are the reverse of the grid';
        is $c->unpdl,    $expected, 'ndarrays in, ndarray out';

        my $out = [ map { [ (0) x 6 ] } 1 .. 4 ];
        is scalar $r->run($a, $nested_b, $out, 5), $expected, 'mixed: ndarray and nested array in, nested array out';
        is scalar($r->run($nested_a, $nested_b, zeroes(float, 6, 4), 5))->unpdl, $expected,
            'nested arrays in, ndarray out';

        my $double_a = pdl(double, $nested_a);
        is scalar($r->run($double_a, $b, zeroes(float, 6, 4), 5))->unpdl, $expected,
            'an input ndarray of another type is converted';
        is [ $double_a->dims ], [ 5, 4 ], 'and left unchanged';

        like dies { $r->run($a, $b, zeroes(double, 6, 4), 5) }, qr/ndarray 'c' must be of type float/,
            'a result ndarray must have the declared type';
        like dies { $r->run($a->flat, $b, zeroes(float, 6, 4), 5) }, qr/ndarray 'a' must have 2 dimensions/,
            'rank is checked';
    };
}

subtest 'empty ndarray for the result' => sub {
    for my $engine (ndarray_engines()) {
        my $r = PerCUDA->new(
            engine   => $engine,
            argument => [ in => 'd', out => 'd' ],
            retval   => ['out'],
            source   => '$out->[$x] = $in->[$x] / 2;',
        );
        my $out = zeroes(double, 0);
        $r->run(sequence(double, 4), $out);
        is $out->unpdl, [ 0, 0.5, 1, 1.5 ], "$engine: reshaped to the size argument and filled";
    }
};

subtest 'device arrays from and to ndarrays' => sub {
    for my $engine (ndarray_engines()) {
        my $r = PerCUDA->new(
            engine   => $engine,
            dim      => 2,
            argument => [ in => 'f', out => 'f' ],
            retval   => ['out'],
            source   => '$out->[$x][$y] = $in->[$x][$y] * 2;',
        );
        my $in  = $r->array(sequence(float, 3, 2), type => 'f');
        my $out = $r->array(type => 'f', dims => [ 2, 3 ]);
        is $in->dims, [ 2, 3 ], "$engine: dims of an ndarray, outermost first";
        $r->run($in, $out);
        my $nd = zeroes(float, 0);
        $out->get(into => $nd);
        is [ $nd->dims ], [ 3, 2 ], "$engine: get(into => \$ndarray) reshapes it";
        is $nd->unpdl, [ [ 0, 2, 4 ], [ 6, 8, 10 ] ], "$engine: and fills it";
    }
};

subtest 'size argument as an ndarray' => sub {
    for my $engine (ndarray_engines()) {
        my $r = PerCUDA->new(
            engine   => $engine,
            dim      => 2,
            size     => 'grid',
            argument => [ grid => 'i', out => 'i' ],
            retval   => ['out'],
            source   => '$out->[$x][$y] = $x * 10 + $y + $grid->[$x][$y];',
        );
        my $grid = ones(long, 3, 2);
        is $r->run($grid, [])->[1], [ 11, 12, 13 ], "$engine: grid shape from ->dims (3, 2)";
    }
};

done_testing;
