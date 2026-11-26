use v5.40;
use Test2::V0;
use lib 't/lib';
use PerCUDATest;

my @warnings;
local $SIG{__WARN__} = sub { push @warnings, @_ };
use PerCUDA;
use PerCUDA::Parser;
use PerCUDA::Analyzer qw(index_dim);

subtest 'dim is the number of variables the kernel declares' => sub {
    my sub dim ($source) { index_dim(PerCUDA::Parser->new(source => $source)->parse) }
    is dim('my $x = thread_index(); $a->[$x] = 1;'),                    1,     'one variable';
    is dim('my ($x, $y) = thread_index(); $a->[$x][$y] = 1;'),          2,     'two variables';
    is dim('my ($i, $j, $k) = thread_index(); $a->[$i][$j][$k] = 1;'),  3,     'three variables, any names';
    is dim('my ($n, $m) = grid_size(); $a->[0][0] = $n;'),              2,     'grid_size alone gives the dimension';
    is dim('my ($x, $y) = thread_index(); my ($n, $m) = grid_size();'), 2,     'both agree';
    is dim('my $y = 1; $a->[0] = $y;'),                                 undef, 'a local $y declares nothing';
    is dim('for my $y (0 .. 3) { $a->[0] += $y; }'),                    undef, 'a loop variable declares nothing';
    is dim('$a->[0] = 1;'),                                             undef, 'no declaration';
};

for my $engine (both_engines) {
    subtest "$engine engine" => sub {
        my $axpy = PerCUDA->new(
            engine   => $engine,
            argument => [ a => 'f', b => 'f', out => 'f', s => 'f' ],
            source   => 'my ($x, $y) = thread_index(); $out->[$x][$y] = $a->[$x][$y] * $s + $b->[$x][$y];',
        );
        is $axpy->engine, $engine, 'engine';

        # dim 2 (inferred), retval out (inferred), size a
        my $a = [ [ 1, 2 ], [ 3, 4 ] ];
        my $b = [ [ 1, 1 ], [ 1, 1 ] ];
        is scalar $axpy->run($a, $b, [], 2), [ [ 3, 5 ], [ 7, 9 ] ], 'positional run returns the written array';
        is scalar $axpy->run(s => 2, b => $b, a => $a), [ [ 3, 5 ], [ 7, 9 ] ],
            'named run in any order; an array never read may be omitted';
        my $out = [];
        $axpy->run(a => $a, b => $b, out => $out, s => 1);
        is $out, [ [ 2, 3 ], [ 4, 5 ] ], 'a named retval array is filled in place';

        like dies { $axpy->run(a => $a, s => 2) },  qr/missing argument 'b'/, 'a read array is required';
        like dies { $axpy->run(a => $a, b => $b) }, qr/missing argument 's'/, 'a scalar is required';
        like dies { $axpy->run(a => $a, t => 1) },  qr/unknown argument 't'/, 'unknown name';
        like dies { $axpy->run(a => $a, a => $a) }, qr/argument 'a' twice/,   'twice';
        like dies { $axpy->run(a => $a, b => $b, 's') }, qr/name => value pairs/, 'odd list';

        my $quiet = PerCUDA->new(
            engine   => $engine,
            argument => [ a => 'f', out => 'f' ],
            retval   => [],
            source   => 'my $x = thread_index(); $out->[$x] = $a->[$x];'
        );
        is [ $quiet->run([1], []) ], [], 'retval => [] returns nothing';

        subtest 'the default size comes from an array passed non-empty' => sub {
            my $r = PerCUDA->new(
                engine   => $engine,
                argument => [ out => 'f', in => 'f' ],
                source   => 'my $x = thread_index(); $out->[$x] = $in->[$x] * 2;',
            );
            is scalar $r->run([], [ 1, 2, 3 ]),          [ 2, 4, 6 ], 'an empty output before the input';
            is scalar $r->run(in => [ 1, 2, 3 ]),        [ 2, 4, 6 ], 'named, without the output';
            is scalar $r->run([ 9, 9, 9, 9 ], [ 1, 2 ]), [ 2, 4, 9, 9 ], 'an array the kernel reads comes first';
        };

        subtest 'the default size is the first array with dim dimensions' => sub {
            my $r = PerCUDA->new(
                engine   => $engine,
                argument => [ w => 'f', m => 'f' ],
                source   => 'my ($x, $y) = thread_index(); $m->[$x][$y] = $w->[$y];',
            );
            is scalar $r->run(w => [ 1, 2, 3 ], m => [ [ 0, 0, 0 ], [ 0, 0, 0 ] ]), [ [ 1, 2, 3 ], [ 1, 2, 3 ] ],
                'size m, not w';
        };

        subtest 'constants' => sub {
            my $r = PerCUDA->new(
                engine    => $engine,
                precision => 'double',
                argument  => [ out => 'd', k => 'i' ],
                constant  => { PI => 3.14159265358979, N => -3, H => 1.5e-3, BIG => 2**40 },
                size      => [4],
                source    => 'my $x = thread_index(); $out->[$x] = $PI * $x + $N * $k + $H + $BIG + ($N % 2);',
            );
            my $want = [ map { 3.14159265358979 * $_ + -3 * 2 + 1.5e-3 + 2**40 + (-3 % 2) } 0 .. 3 ];
            is scalar $r->run(k => 2), $want, 'values, negative ints and exponents';
            like dies {
                PerCUDA->new(
                    engine   => $engine,
                    argument => [ a => 'f' ],
                    constant => { K => 1 },
                    source   => 'my $x = thread_index(); $K = 2; $a->[$x] = 1;'
                )
            }, qr/constant \$K cannot be assigned/, 'assignment';
            like dies {
                PerCUDA->new(
                    engine   => $engine,
                    argument => [ a => 'f' ],
                    constant => { K => 1 },
                    source   => 'my $x = thread_index(); my $K = 2; $a->[$x] = $K;'
                )
            }, qr/constant \$K cannot be declared/, 'declaration';
            like dies {
                PerCUDA->new(
                    engine   => $engine,
                    argument => [ a => 'f' ],
                    constant => { a => 1 },
                    source   => 'my $x = thread_index(); $a->[$x] = 1;'
                )
            }, qr/constant 'a' is also an argument/, 'clash';
            like dies {
                PerCUDA->new(
                    engine   => $engine,
                    argument => [ a => 'f' ],
                    constant => { K => 'inf' },
                    source   => 'my $x = thread_index(); $a->[$x] = $K;'
                )
            }, qr/constant 'K' must be a finite number/, 'inf';
        };

        subtest 'an explicit grid' => sub {
            my $r = PerCUDA->new(
                engine   => $engine,
                argument => [ out => 'i', n => 'i', seed => 'i' ],
                size     => [ 'n', 3 ],
                source   => 'my ($x, $y) = thread_index(); $out->[$x][$y] = $x * 10 + $y + $seed;',
            );
            is scalar $r->run(n => 2, seed => 1), [ [ 1, 2, 3 ], [ 11, 12, 13 ] ], 'grid from a scalar and an integer';
            like dies { $r->run(n => -1, seed => 1) }, qr/size 'n' must be a non-negative integer/, 'negative size';
            like dies {
                PerCUDA->new(
                    engine   => $engine,
                    argument => [ out => 'i', n => 'f' ],
                    size     => ['n'],
                    source   => 'my $x = thread_index(); $out->[$x] = $n;'
                )
            }, qr/must be an int argument/, 'float size';
            like dies {
                PerCUDA->new(
                    engine   => $engine,
                    argument => [ out => 'i' ],
                    size     => [ 2, 2 ],
                    source   => 'my $x = thread_index(); $out->[$x] = 1;'
                )
            }, qr/size gives 2 dimension\(s\) but thread_index\(\) gives 1/,
                'size list and declared variables must agree';
            like dies {
                PerCUDA->new(
                    engine   => $engine,
                    argument => [ out => 'i' ],
                    size     => [ 2, 2 ],
                    'dim'    => 2,
                    source   => 'my ($x, $y) = thread_index(); $out->[$x][$y] = 1;'
                )
            }, qr/dim is not an option/, 'the dim option is gone';
            is scalar PerCUDA->new(
                engine   => $engine,
                argument => [ out => 'i' ],
                size     => [ 2, 2 ],
                source   => '$out->[1][1] = 7;'
                )->run([]),
                [ [ 0, 0 ], [ 0, 7 ] ], 'without a declaration, the length of the size list is the dimension';
            is scalar $r->run(n => 0, seed => 1), [], 'an empty grid';

            my $named = PerCUDA->new(
                engine   => $engine,
                argument => [ out => 'i', n => 'i' ],
                size     => 'n',
                source   => 'my $x = thread_index(); $out->[$x] = $n - $x;',
            );
            is scalar $named->run(n => 3), [ 3, 2, 1 ], "size => 'n' with an integer scalar";

            my $jobs = PerCUDA->new(
                engine    => $engine,
                perl_jobs => 2,
                argument  => [ out => 'i', n => 'i' ],
                size      => ['n'],
                source    => 'my $x = thread_index(); $out->[$x] = $x * $x;',
            );
            is scalar $jobs->run(n => 7), [ map { $_ * $_ } 0 .. 6 ], 'with perl_jobs';
        };

        subtest 'positional run with a non-finite first scalar' => sub {
            my $r = PerCUDA->new(
                engine    => $engine,
                argument  => [ s => 'd', out => 'd' ],
                precision => 'double',
                source    => 'my $x = thread_index(); $out->[$x] = $s;'
            );
            my $inf = 9**9**9;
            is scalar $r->run($inf, [ 0, 0 ]), [ $inf, $inf ], 'Inf is a value, not a name';
        };

        subtest 'single precision constants' => sub {
            my $r = PerCUDA->new(
                engine   => $engine,
                argument => [ out => 'f' ],
                constant => { C => 0.1, D => '1e-7' },
                size     => [3],
                source   => 'my $x = thread_index(); $out->[$x] = $C * $x + $D;',
            );
            my $f32 = sub ($v) { unpack 'f', pack 'f', $v };
            is scalar $r->run([]), [ map { $f32->($f32->($f32->(0.1) * $_) + $f32->(1e-7)) } 0 .. 2 ],
                'rounded like single precision literals';
        };

        subtest 'type names' => sub {
            my $r = PerCUDA->new(
                engine   => $engine,
                argument => [ in => 'int', out => 'double', s => 'float' ],
                source   => 'my $x = thread_index(); $out->[$x] = $in->[$x] / 2 + $s;',
            );
            is scalar $r->run([ 1, 2 ], [], 0.5), [ 1, 1.5 ], 'int, float and double';
            my $mixed = PerCUDA->new(
                engine   => $engine,
                argument => [ in => 'i', out => 'double' ],
                source   => 'my $x = thread_index(); $out->[$x] = $in->[$x];'
            );
            is scalar $mixed->run([ 3, 4 ], []), [ 3, 4 ], 'short codes still work, mixed with names';
            my $a = $r->array(type => 'double', dims => [2]);
            is $a->type,                                  'double', 'an array reports its type name';
            is $r->array(type => 'i', dims => [1])->type, 'int',    'made with a short code';
            is $r->array(dims => [1])->type,              'float',  'an array without a type is a float';
            is $r->array([ 1, 2 ])->type,                 'float',  'also from data';
            is(PerCUDA->context(engine => $engine, precision => 'double')->array(dims => [1])->type,
                'double', 'or a double with precision => double');
            like dies { $r->run($r->array(type => 'float', dims => [2]), [], 1) },
                qr/'in' must be of type int, not float/, 'messages use names';
            like dies {
                PerCUDA->new(
                    engine   => $engine,
                    argument => [ a => 'long' ],
                    source   => 'my $x = thread_index(); $a->[$x] = 1;'
                )
            }, qr/unknown type 'long' \(use int, float or double\)/, 'unknown type';
            like dies { $r->array(type => 'long', dims => [1]) }, qr/must be 'int', 'float' or 'double'/,
                'unknown array type';
        };

        subtest 'context' => sub {
            my $pc = PerCUDA->context(engine => $engine, precision => 'double');
            is $pc->engine, $engine, 'engine';
            my $scale = $pc->kernel(
                argument => [ u => 'd', v => 'd', s => 'd' ],
                source   => 'my $x = thread_index(); $v->[$x] = $u->[$x] * $s;'
            );
            my $u = $pc->array([ 1, 2, 3 ], type => 'd');
            my $v = $pc->array(type => 'd', dims => [3]);
            ref_is scalar $scale->run(u => $u, v => $v, s => 0.5), $v, 'a context array is used in place';
            is $v->get, [ 0.5, 1, 1.5 ], 'result';
            is $v->sum, 3,               'reduction of a context array';
            like $scale->cuda_source, qr/double/, 'the defaults reach the kernel' if $engine eq 'gpu';
            like dies { $pc->kernel(engine => 'perl', argument => [], source => '') }, qr/'engine' is fixed/,
                'engine cannot be overridden';

            my $cc = PerCUDA->context(engine => $engine, constant => { A => 1 });
            my $k  = $cc->kernel(
                argument => [ out => 'i' ],
                constant => { B => 2 },
                size     => [2],
                source   => 'my $x = thread_index(); $out->[$x] = $A + $B;'
            );
            is scalar $k->run([]), [ 3, 3 ], 'kernel constants add to the context constants';
        };
    };
}

is \@warnings, [], 'no warnings';

done_testing;
