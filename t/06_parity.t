use v5.40;
use Test2::V0;
use lib 't/lib';
use PerCUDATest;
use PerCUDA;

# The perl engine follows the GPU semantics, so these must agree exactly.
my %CASE = (
    'bit operations on negative numbers' => <<'K',
my $v = $in->[$x];
$out->[$x] = (($v & -6) | ($v ^ 3)) + ($v << 3) + ($v >> 2) + ~$v;
K
    'comparisons and not yield 0/1' => <<'K',
my $v = $in->[$x];
$out->[$x] = ($v < 0) * 1000 + ($v == 3) * 100 + (!$v) * 10 + ($v >= 7);
K
    'stores into int arrays truncate' => <<'K',
$out->[$x] = $in->[$x] / 4;
K
    'integers wrap around at 32 bits' => <<'K',
my $v = $in->[$x] * 100003;
my $w = 2147483600 + $in->[$x];
$w += 100;
$w++;
$out->[$x] = $v * $v + -$w + 3 ** 25;
K
    'modulus by zero is 0' => <<'K',
my $d = $in->[$x] % 3;
$out->[$x] = 7 % $d + $in->[$x] % ($d - $d);
K
);

subtest 'division by zero gives inf and nan' => sub {
    my %result;
    for my $engine (both_engines()) {
        my $r = PerCUDA->new(
            engine   => $engine,
            argument => [ in => 'i', out => 'f' ],
            retval   => ['out'],
            source   => 'my $d = $in->[$x] % 3; $out->[$x] = $in->[$x] / $d;',
        );
        $result{$engine} = $r->run([ -4 .. 4 ], []);
    }
    is $result{perl}, [qw(-2 -Inf -2 -0.5 NaN 1 1 Inf 4)], 'perl';
    is $result{gpu},  $result{perl},                       'perl and gpu agree' if $result{gpu};
};

subtest 'perl_mode => fast' => sub {
    my %args = (
        engine   => 'perl',
        argument => [ in => 'f', out => 'f' ],
        retval   => ['out'],
        source   => 'my $s = 0.0; for my $i (1 .. 100) { $s += $in->[$x] / $i; } $out->[$x] = $s;',
    );
    my $compat = PerCUDA->new(%args);
    my $fast   = PerCUDA->new(%args, perl_mode => 'fast');
    like $compat->perl_source, qr/pack\('f'/, 'compat rounds to single precision';
    unlike $fast->perl_source, qr/pack\('f'/, 'fast does not';
    my ($c, $f) = ($compat->run([ 1 .. 8 ], []), $fast->run([ 1 .. 8 ], []));
    ok !grep({ abs($c->[$_] - $f->[$_]) > 1e-5 * abs $c->[$_] } 0 .. 7), 'results are close';
    like dies { PerCUDA->new(%args, perl_mode => 'slow') }, qr/perl_mode must be/, 'rejects unknown perl_mode';
};

for my $name (sort keys %CASE) {
    subtest $name => sub {
        my %result;
        for my $engine (both_engines()) {
            my $r = PerCUDA->new(
                engine   => $engine,
                argument => [ in => 'i', out => 'i' ],
                retval   => ['out'],
                source   => $CASE{$name},
            );
            $result{$engine} = $r->run([ -40 .. 40 ], []);
        }
        ok !grep({ $_ != int $_ } @{ $result{perl} }), 'perl results are integers';
        is $result{gpu}, $result{perl}, 'perl and gpu agree exactly' if $result{gpu};
    };
}

subtest 'arrays outside retval are not modified' => sub {
    for my $engine (both_engines()) {
        my $r = PerCUDA->new(
            engine   => $engine,
            argument => [ in => 'i', tmp => 'i', unused => 'i', out => 'i' ],
            retval   => ['out'],
            source   => '$tmp->[$x] = $in->[$x] * 2; $out->[$x] = $tmp->[$x] + 1 + $unused->[$x];',
        );
        my ($in, $tmp, $unused) = ([ 1 .. 4 ], [ (9) x 4 ], []);
        is scalar $r->run($in, $tmp, $unused, []), [ 3, 5, 7, 9 ], "$engine: result";
        is $tmp,                                   [ (9) x 4 ],    "$engine: written non-retval array is unchanged";
        is $unused,                                [],             "$engine: empty input array is unchanged";
    }
};

subtest 'empty output is zero filled' => sub {
    for my $engine (both_engines()) {
        my $r = PerCUDA->new(
            engine   => $engine,
            argument => [ in => 'i', out => 'i' ],
            size     => 'in',
            retval   => ['out'],
            source   => '$out->[$x] = 1 if $x % 2;',
        );
        is scalar $r->run([ 1 .. 4 ], []), [ 0, 1, 0, 1 ], $engine;
    }
};

subtest 'elements the kernel does not write keep their values' => sub {
    my %covered = (
        'every element written'      => [ '$out->[$x] = $x;',                   [ 0, 1,  2,  3 ] ],
        'return before the store'    => [ 'return if $x % 2; $out->[$x] = $x;', [ 0, 9,  2,  9 ] ],
        'store under a condition'    => [ 'if ($x > 1) { $out->[$x] = $x; }',   [ 9, 9,  2,  3 ] ],
        'store reads the old value'  => [ '$out->[$x] += $x;',                  [ 9, 10, 11, 12 ] ],
        'store into another element' => [ '$out->[3 - $x] = $x if $x < 2;',     [ 9, 9,  1,  0 ] ],
    );
    for my $name (sort keys %covered) {
        my ($source, $expected) = @{ $covered{$name} };
        for my $engine (both_engines()) {
            my $r = PerCUDA->new(
                engine   => $engine,
                argument => [ in => 'i', out => 'i' ],
                size     => 'in',
                retval   => ['out'],
                source   => $source,
            );
            is scalar $r->run([ 1 .. 4 ], [ (9) x 4 ]), $expected, "$name ($engine)";
        }
    }
};

subtest 'precision => double' => sub {
    my $source = <<'K';
my $s = 0.0;
for my $i (1 .. 1000) { $s += 0.1 * $in->[$x]; }
$out->[$x] = $s;
K
    my %result;
    for my $engine (both_engines()) {
        for my $precision (qw(single double)) {
            my $r = PerCUDA->new(
                engine    => $engine,
                precision => $precision,
                argument  => [ in => 'f', out => 'd' ],
                retval    => ['out'],
                source    => $source,
            );
            like $r->cuda_source, $precision eq 'double' ? qr/double l\d+_s/ : qr/float l\d+_s/, "$precision local";
            $result{$engine}{$precision} = $r->run([ 1 .. 8 ], []);
        }
    }
    my $perl = $result{perl}{double};
    if (my $gpu = $result{gpu}) {
        my $max_err = sub ($got) {
            my $m = 0;
            $m = abs($got->[$_] - $perl->[$_]) > $m ? abs($got->[$_] - $perl->[$_]) : $m for 0 .. 7;
            $m;
        };
        is $gpu->{single}, $result{perl}{single}, 'single: perl and gpu agree exactly';
        is $gpu->{double}, $result{perl}{double}, 'double: perl and gpu agree exactly';
        cmp_ok $max_err->($gpu->{single}), '>', 1e-6, 'single drifts from double';
    }
    like dies {
        PerCUDA->new(engine => 'perl', precision => 'half', argument => [ a => 'f' ], source => '$a->[$x] = 1;')
    }, qr/precision must be/, 'rejects unknown precision';
};

done_testing;
