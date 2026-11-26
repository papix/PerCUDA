use v5.40;
use Test2::V0;
use lib 't/lib';
use PerCUDATest;
use PerCUDA;

my $grid = [ [ 1, 2, 3 ], [ 4, 5, 6 ] ];    # 2 x 3

my sub runner ($engine, $source, %args) {
    return PerCUDA->new(
        engine   => $engine,
        dim      => 2,
        check    => 1,
        size     => 'in',
        argument => [ in => 'f', out => 'f' ],
        retval   => ['out'],
        source   => $source,
        %args,
    );
}

my @bad = (
    [
        'read past the end of dimension 0',
        '$out->[$x][$y] = $in->[$x + 1][$y];',
        qr/\$in, dimension 0, index 2 is not in 0 \.\. 1 \(thread \$x=1, \$y=\d\)/
    ],
    [
        'read before the start of dimension 1',
        '$out->[$x][$y] = $in->[$x][$y - 1];',
        qr/\$in, dimension 1, index -1 is not in 0 \.\. 2 \(thread \$x=\d, \$y=0\)/
    ],
    [ 'write out of range',     '$out->[$x][$y + 1] = 1.0;',  qr/\$out, dimension 1, index 3 is not in 0 \.\. 2/ ],
    [ 'read and write with +=', '$out->[$x + 2][$y] += 1.0;', qr/\$out, dimension 0, index [23] is not in 0 \.\. 1/ ],
    [
        'line number of a later line',
        "my \$s = 0.0;\nfor my \$i (0 .. 3) {\n    \$s += \$in->[\$x][\$i];\n}\n\$out->[\$x][\$y] = \$s;",
        qr/at kernel line 3: \$in, dimension 1, index 3 is not in 0 \.\. 2/
    ],
);

for my $engine (both_engines()) {
    subtest $engine => sub {
        for my $case (@bad) {
            my ($name, $source, $re) = @$case;
            like dies { runner($engine, $source)->run($grid, []) }, $re, $name;
        }
        is scalar runner($engine, '$out->[$x][$y] = $in->[$x][$y] * 2;')->run($grid, []),
            [ [ 2, 4, 6 ], [ 8, 10, 12 ] ],
            'in range: no error';
        is scalar runner($engine, '$out->[$x][$y] = $in->[$x * 0.5][$y - 0.5];')->run($grid, []),
            [ [ 1, 1, 2 ], [ 1, 1, 2 ] ], 'indices are truncated toward zero first, as in C (-0.5 is 0)';
        my $stencil = <<'K';
my $s = $in->[$x][$y];
$s += $in->[$x - 1][$y] if $x > 0;
$s += $in->[$x + 1][$y] if $x < $size_x - 1;
$s += $in->[$x][$y - 1] if $y > 0;
$s += $in->[$x][$y + 1] if $y < $size_y - 1;
$out->[$x][$y] = $s;
K
        is scalar runner($engine, $stencil)->run($grid, []), [ [ 7, 11, 11 ], [ 10, 17, 14 ] ],
            'guarded stencil: no error';

        my $r   = runner($engine, '$out->[$x][$y] = $in->[$x][$y + 1];');
        my $dev = $r->array($grid, type => 'f');
        like dies { $r->run($dev, []) }, qr/\$in, dimension 1, index 3/, 'device arrays are checked';

        my $cube = runner($engine, '$out->[$x][$y][$z] = $in->[$x][$y][$z + $size_z];', dim => 3);
        like dies { $cube->run([ [ [ 1, 2 ] ] ], []) },
            qr/\$in, dimension 2, index 2 is not in 0 \.\. 1 \(thread \$x=0, \$y=0, \$z=0\)/,
            '3 dimensions';
    };
}

like dies { runner('perl', '$out->[$x][$y] = $in->[$x + 1][$y];', perl_jobs => 2)->run($grid, []) },
    qr/index out of range/,
    'perl_jobs: errors of worker processes are reported';

like dies { runner('perl', '$out->[$x][$y] = $in->[$x + 1][$y];')->run($grid, []) }, qr/at \Q${\__FILE__}\E line \d+/,
    'the error points at the caller of run';

my $plain = PerCUDA->new(
    engine   => 'perl',
    argument => [ in => 'f', out => 'f' ],
    retval   => ['out'],
    source   => '$out->[$x] = $in->[$x];'
);
unlike $plain->cuda_source, qr/percuda_at|percuda_err/, 'without check, the generated code has no checks';
unlike $plain->perl_source, qr/Runtime::at/,            'nor the perl code';

done_testing;
