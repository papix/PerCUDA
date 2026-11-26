use v5.40;
use Test2::V0;
use lib 't/lib';
use PerCUDATest;
use PerCUDA;

# Kernels read the thread index with my (...) = thread_index() and the grid
# size with my (...) = grid_size(); nothing is declared for them.

package Thread::Kernel {
    use PerCUDA;

    sub table : GPU(int, { engine => 'perl' }) ($out) {
        my ($x, $y) = thread_index();
        $out->[$x][$y] = $x * 10 + $y;
    }
}

package Own::Grid {
    sub grid_size { return 'own' }
    use PerCUDA;
}

my sub run_kernel ($engine, $source, $argument, @values) {
    my $runner = PerCUDA->new(engine => $engine, source => $source, argument => $argument);
    return $runner->run(@values);
}

for my $engine (both_engines()) {
    subtest $engine => sub {
        is run_kernel($engine, 'my $i = thread_index(); $out->[$i] = $i * 2;', [ out => 'int' ], [ (0) x 4 ]),
            [ 0, 2, 4, 6 ], 'scalar form: 1 dimension';

        is run_kernel(
            $engine,
            'my ($row, $col) = thread_index; $out->[$row][$col] = $row * 10 + $col;',
            [ out => 'int' ],
            [ [ 0, 0, 0 ], [ 0, 0, 0 ] ]
            ),
            [ [ 0, 1, 2 ], [ 10, 11, 12 ] ], 'any names, no parentheses on the call';

        is run_kernel(
            $engine,
            'my ($i, $j, $k) = thread_index(); my ($ni, $nj, $nk) = grid_size();'
                . ' $out->[$i][$j][$k] = (($i * $nj) + $j) * $nk + $k;',
            [ out => 'int' ],
            [
                map {
                    [ map { [ 0, 0 ] } 1 .. 3 ]
                } 1 .. 2
            ]
            ),
            [ [ [ 0, 1 ], [ 2, 3 ], [ 4, 5 ] ], [ [ 6, 7 ], [ 8, 9 ], [ 10, 11 ] ] ], '3 dimensions with grid_size';

        is run_kernel(
            $engine,
            'my ($n) = grid_size(); my ($i) = thread_index(); $out->[$i] = $n - $i;',
            [ out => 'int' ],
            [ (0) x 3 ]
            ),
            [ 3, 2, 1 ], 'grid_size before thread_index';

        is run_kernel(
            $engine,
            'my $x = thread_index(); { my $x = 7; $out->[0] = $x if 0; } $out->[$x] = $x;',
            [ out => 'int' ],
            [ (9) x 2 ]
            ),
            [ 0, 1 ], 'an inner my shadows the index';

        # Each thread has its own copies: assigning them changes no other thread.
        my $grid = [ [ 0, 0, 0 ], [ 0, 0, 0 ] ];
        is run_kernel(
            $engine,
            'my ($i, $j) = thread_index(); $out->[$i][$j] = $i * 10 + $j; $i = 0; $j = 0;',
            [ out => 'int' ], $grid
            ),
            [ [ 0, 1, 2 ], [ 10, 11, 12 ] ], 'assigning the thread index stays in the thread';
        is run_kernel(
            $engine,
            'my ($i, $j) = thread_index(); my ($ni, $nj) = grid_size(); $out->[$i][$j] = $ni * 10 + $nj; $ni = 1; $nj = 1;',
            [ out => 'int' ],
            $grid
            ),
            [ [ 23, 23, 23 ], [ 23, 23, 23 ] ], 'assigning the grid size stays in the thread';
        is run_kernel(
            $engine,
            'my ($i, $j) = thread_index(); $j++; $out->[$i][ $j - 1 ] = $i * 10 + $j;',
            [ out => 'int' ], $grid
            ),
            [ [ 1, 2, 3 ], [ 11, 12, 13 ] ], 'the index can be incremented';

        my $runner = PerCUDA->new(
            engine   => $engine,
            check    => 1,
            argument => [ out => 'int' ],
            source   => 'my ($r, $c) = thread_index(); $out->[$r][$c + 1] = 1;',
        );
        like dies { $runner->run([ [ 0, 0 ], [ 0, 0 ] ]) },
            qr/\$out, dimension 1, index 2 is not in 0 \.\. 1 \(thread \$r=\d, \$c=1\)/,
            'check names the declared variables';

        my $plain = PerCUDA->new(
            engine   => $engine,
            check    => 1,
            size     => [ 2, 2 ],
            argument => [ out => 'int' ],
            source   => '$out->[5] = 1;',
        );
        like dies { $plain->run([0]) }, qr/index 5 is not in 0 \.\. 0 \(thread \d, \d\)/,
            'check without declared names';
    };
}

subtest 'kernel subs' => sub {
    is Thread::Kernel::table([ [ 0, 0 ], [ 0, 0 ] ]), [ [ 0, 1 ], [ 10, 11 ] ], 'a :GPU sub compiles under strict';
    like dies { Thread::Kernel::thread_index() }, qr/thread_index\(\) can only be called in a kernel/,
        'thread_index outside a kernel dies';
    like dies { Thread::Kernel::grid_size() }, qr/grid_size\(\) can only be called in a kernel/,
        'grid_size outside a kernel dies';
    ok !exists $Thread::Kernel::{size_x}, 'no $size_x is exported';
    is Own::Grid::grid_size(), 'own', 'a sub of the package by the same name is kept';
};

my @rejected = (
    [ '$out->[$x] = 1;',                                     qr/variable \$x is not declared .*thread_index\(\)/ ],
    [ '$out->[0] = $size_x;',                                qr/variable \$size_x is not declared .*grid_size\(\)/ ],
    [ '$out->[thread_index()] = 1;',                         qr/thread_index\(\) can only be used as my/ ],
    [ 'my $i = thread_index() + 1; $out->[$i] = 1;',         qr/thread_index\(\) can only be used as my/ ],
    [ 'my ($i, $j);',                                        qr/my \(\.\.\.\) is only supported as/ ],
    [ 'if (1) { my ($i) = thread_index(); }',                qr/must be a top-level statement/ ],
    [ 'my ($i) = thread_index(); my ($j) = thread_index();', qr/thread_index\(\) is called twice/ ],
    [ 'my ($i, $i) = thread_index();',                       qr/\$i is declared twice/ ],
    [ 'my ($a, $b, $c, $d) = thread_index();',               qr/gives at most 3 values/ ],
    [
        'my ($i, $j) = thread_index(); my ($n) = grid_size();',
        qr/thread_index\(\) gives 2 value\(s\) but grid_size\(\) gives 1/
    ],
    [ 'my ($out) = thread_index(); $out = 1;',     qr/\$out has the name of an argument/ ],
    [ '$out->[$i] = 1; my ($i) = thread_index();', qr/variable \$i is not declared/ ],
);
for my $case (@rejected) {
    my ($source, $error) = @$case;
    like dies { PerCUDA->new(engine => 'perl', argument => [ out => 'int' ], source => $source) }, $error,
        "rejects: $source";
}

like dies {
    PerCUDA->new(engine => 'perl', dim => 2, argument => [ out => 'int' ], source => 'my ($i, $j) = thread_index();')
}, qr/dim is not an option/, 'dim is gone';

like dies {
    PerCUDA->new(
        engine   => 'perl',
        size     => [ 2, 2 ],
        argument => [ out => 'int' ],
        source   => 'my $i = thread_index(); $out->[$i] = 1;'
    )
}, qr/size gives 2 dimension\(s\) but thread_index\(\) gives 1/, 'size must match thread_index';

like dies {
    PerCUDA->new(
        engine   => 'perl',
        size     => [2],
        argument => [ out => 'int' ],
        source   => 'my ($ni, $nj) = grid_size(); $out->[0][0] = $ni;'
    )
}, qr/size gives 1 dimension\(s\) but grid_size\(\) gives 2/, 'the message names grid_size when only it is declared';

like dies {
    PerCUDA->new(
        engine   => 'perl',
        size     => 'n',
        argument => [ out => 'int', n => 'int' ],
        source   => 'my ($ni, $nj) = grid_size(); $out->[0][0] = $ni;'
    )->run([], 3)
}, qr/size 'n' gives 1 dimension but grid_size\(\) gives 2/, 'so does the message of a numeric size';

like dies {
    PerCUDA->new(
        engine   => 'perl',
        size     => 'n',
        argument => [ out => 'int', n => 'int' ],
        source   => 'my ($ni, $nj) = grid_size(); my ($i, $j) = thread_index(); $out->[$i][$j] = $ni;'
    )->run([], 3)
}, qr/size 'n' gives 1 dimension but thread_index\(\) gives 2/, 'thread_index is named when both are declared';

is PerCUDA->new(
    engine   => 'perl',
    size     => [ 2, 3 ],
    argument => [ out => 'int' ],
    source   => '$out->[0] += 1;',
)->run([0]), [6], 'without thread_index the size list gives the dimensions';

done_testing;
