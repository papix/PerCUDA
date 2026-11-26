use v5.40;
use Test2::V0;
use lib 't/lib';
use PerCUDATest;
use PerCUDA;

my $LIFE = <<'K';
my $n = 0;
for my $dx (-1 .. 1) {
    for my $dy (-1 .. 1) {
        next if $dx == 0 && $dy == 0;
        $n += $cur->[($x + $dx + $size_x) % $size_x][($y + $dy + $size_y) % $size_y];
    }
}
$next->[$x][$y] = $n == 3 || ($n == 2 && $cur->[$x][$y]) ? 1 : 0;
K

my sub life ($engine, %args) {
    return PerCUDA->new(
        engine   => $engine,
        dim      => 2,
        size     => 'cur',
        argument => [ cur => 'i', next => 'i' ],
        retval   => ['next'],
        source   => $LIFE,
        %args,
    );
}

my $glider = [ map { [ (0) x 8 ] } 1 .. 6 ];
$glider->[0][1] = $glider->[1][2] = $glider->[2][0] = $glider->[2][1] = $glider->[2][2] = 1;

# Reference: 12 steps with nested arrays on the perl engine.
my $expected = do {
    my $r = life('perl');
    my ($cur, $next) = ([ map { [@$_] } @$glider ], []);
    for (1 .. 12) { $r->run($cur, $next); ($cur, $next) = ($next, $cur) }
    $cur;
};

for my $engine (both_engines()) {
    subtest $engine => sub {
        my $r    = life($engine);
        my $cur  = $r->array($glider, type => 'i');
        my $next = $r->array(type => 'i', dims => [ 6, 8 ]);
        is $cur->dims,   [ 6, 8 ],                       'dims, outermost first';
        is $cur->type,   'int',                          'type';
        is $cur->engine, $engine,                        'engine';
        is $next->get,   [ map { [ (0) x 8 ] } 1 .. 6 ], 'dims => [...] makes zeros';

        my $result = $r->run($cur, $next);
        ref_is $result, $next, 'run returns the device array given for the retval';
        ($cur, $next) = ($next, $cur);
        for (2 .. 12) { $r->run($cur, $next); ($cur, $next) = ($next, $cur) }
        is $cur->get, $expected, '12 steps on the device match nested arrays';

        my $into = [ [9] ];
        ref_is $cur->get(into => $into), $into, 'get(into => $array) returns that array';
        is $into,           $expected, 'and fills it';
        is $glider->[2][2], 1,         'the source array is left unchanged';

        is scalar $r->run($r->array($glider, type => 'i'), []), life('perl')->run($glider, []),
            'a device input with a nested retval';

        like dies { $r->run($r->array($glider, type => 'f'), $next) }, qr/'cur' must be of type int, not float/,
            'type is checked';
        like dies { $r->run($cur, $r->array([ 1, 2 ], type => 'i')) }, qr/'next' must have 2 dimensions/,
            'rank is checked';
        like dies { $r->run($cur, $cur) }, qr/same array is passed for 'cur' and 'next'/,
            'an array written by the kernel cannot be another argument too';
        is $r->array($glider)->type, 'float', 'without a type: float';
        like dies { $r->array(type => 'i', dims => []) }, qr/needs data or dims/, 'dims are checked';
        like dies { $r->array([], type => 'i') }, qr/non-empty/, 'data must not be empty';
    };
}

subtest 'mixed with nested arrays, and written device arrays outside retval' => sub {
    for my $engine (both_engines()) {
        my $r = PerCUDA->new(
            engine   => $engine,
            argument => [ in => 'f', tmp => 'f', out => 'f' ],
            retval   => ['out'],
            source   => '$tmp->[$x] = $in->[$x] * 2; $out->[$x] = $tmp->[$x] + 1;',
        );
        my $tmp = $r->array(type => 'f', dims => [4]);
        is scalar $r->run([ 1 .. 4 ], $tmp, []), [ 3, 5, 7, 9 ], "$engine: nested retval";
        is $tmp->get, [ 2, 4, 6, 8 ], "$engine: a device array is written in place even outside retval";
    }
};

subtest 'perl_jobs with device arrays' => sub {
    my $r    = life('perl', perl_jobs => 3);
    my $cur  = $r->array($glider, type => 'i');
    my $next = $r->array(type => 'i', dims => [ 6, 8 ]);
    for (1 .. 12) { $r->run($cur, $next); ($cur, $next) = ($next, $cur) }
    is $cur->get, $expected, 'same result';
};

SKIP: {
    skip 'needs both engines', 1 unless gpu_available();
    my $perl_array = life('perl')->array($glider, type => 'i');
    like dies { life('gpu')->run($perl_array, []) }, qr/belongs to the perl engine, not gpu/, 'engines do not mix';
}

done_testing;
