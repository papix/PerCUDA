use v5.40;
use Test::More;
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
);

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
        is_deeply $result{gpu}, $result{perl}, 'perl and gpu agree exactly' if $result{gpu};
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
        is_deeply scalar $r->run($in, $tmp, $unused, []), [ 3, 5, 7, 9 ], "$engine: result";
        is_deeply $tmp, [ (9) x 4 ], "$engine: written non-retval array is unchanged";
        is_deeply $unused, [], "$engine: empty input array is unchanged";
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
        is_deeply scalar $r->run([ 1 .. 4 ], []), [ 0, 1, 0, 1 ], $engine;
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
            like $r->c_source, $precision eq 'double' ? qr/double l\d+_s/ : qr/float l\d+_s/, "$precision local";
            $result{$engine}{$precision} = $r->run([ 1 .. 8 ], []);
        }
    }
    my $perl = $result{perl}{double};
    if (my $gpu = $result{gpu}) {
        my $max_err = sub ($got) { my $m = 0; $m = abs($got->[$_] - $perl->[$_]) > $m ? abs($got->[$_] - $perl->[$_]) : $m for 0 .. 7; $m };
        cmp_ok $max_err->($gpu->{double}), '<', 1e-9, 'double on GPU matches perl';
        cmp_ok $max_err->($gpu->{single}), '>', 1e-6, 'single on GPU drifts from perl';
    }
    eval { PerCUDA->new(engine => 'perl', precision => 'half', argument => [ a => 'f' ], source => '$a->[$x] = 1;') };
    like $@, qr/precision must be/, 'rejects unknown precision';
};

done_testing;
