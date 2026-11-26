use v5.40;
use Test2::V0;
use PerCUDA;

my $runner = PerCUDA->new(
    engine   => 'perl',
    argument => [ a => 'f', out => 'i', n => 'i' ],
    returns  => ['out'],
    source   => <<'K',
my ($x, $y) = thread_index();
my $i = 0;
my $f = 0;
$f = $f + $a->[$x][$y] / 2;
my $q = 7 / 2;
$i = $i + $n;
for (my $k = 0; $k < $n - 1; $k++) { $i += $k; }
for (my $j = 0; $j < $n; $j++) { $j++; }
$out->[$x][$y] = $i + $f + $q;
K
);

subtest 'CUDA' => sub {
    my $c = $runner->cuda_source;
    like $c,
        qr/extern "C" __global__ void percuda_kernel\(const float \*__restrict__ v_a, int \*__restrict__ v_out, int v_n, int percuda_size_x, int percuda_size_y, int d_a_1, int d_out_1\)/,
        'signature: read-only arrays are const, scalars and sizes by value';
    like $c, qr/\bint l\d+_i = 0;/,                                           'int local stays int';
    like $c, qr/\bfloat l\d+_f = /,                                           'local assigned a float becomes float';
    like $c, qr/\bfloat l\d+_q = /,                                           'division yields float';
    like $c, qr/v_a\[\(percuda_x\) \* d_a_1 \+ \(percuda_y\)\]/,              'row-major flattening';
    like $c, qr/int percuda_y = blockIdx\.x \* blockDim\.x \+ threadIdx\.x;/, 'last axis runs along CUDA x (coalesced)';
    like $c, qr/if \(percuda_x >= percuda_size_x \|\| percuda_y >= percuda_size_y\) return;/, 'bounds guard';
};

subtest 'Perl' => sub {
    my $perl = $runner->perl_source;
    like $perl, qr/for my \$l\d+_k \(0 \.\. .*\$v_n - 1.* - 1\) \{/, 'counting loop becomes a range';
    like $perl, qr/for \(; \(\$l\d+_j < \$v_n\); /,                  'loop that assigns its counter stays C style';
};

subtest 'write analysis' => sub {
    my sub analyze ($source) {
        my $ast = PerCUDA::Parser->new(source => 'my ($x, $y) = thread_index(); ' . $source)->parse;

        # the analyzer takes the dimension count that the kernel declares
        return PerCUDA::Analyzer->new(
            ast => $ast,
            'dim', PerCUDA::Analyzer::index_dim($ast), arguments => [ [ a => 'f' ], [ out => 'f' ] ]
        );
    }
    my sub covered ($source) { return analyze($source)->args->[1]{covered} }

    ok covered('my $v = $a->[$x][$y]; $out->[$x][$y] = $v;'), 'unconditional store into the own element';
    ok !covered('$out->[$y][$x] = 1;'),                       'transposed store';
    ok !covered('return if $x; $out->[$x][$y] = 1;'),         'store after a return';
    ok !covered('$out->[$x][$y] = 1 if $y;'),                 'conditional store';
    ok !covered('$out->[$x][$y] = $out->[$x][$y] + 1;'),      'array is read';
    ok !covered('$x = 0; $out->[$x][$y] = 1;'),               'thread index reassigned';

    ok analyze('$out->[$x][0] = 1; $out->[$x][$y] += 2;')->writes_own_rows, 'all stores use $x first';
    ok !analyze('$out->[$y][$x] = 1;')->writes_own_rows,                    'store into another row';
};

done_testing;
