use v5.40;
use Test::More;
use PerCUDA;

my $runner = PerCUDA->new(
    engine   => 'perl',
    dim      => 2,
    argument => [ a => 'f', out => 'i', n => 'i' ],
    retval   => ['out'],
    source   => <<'K',
my $i = 0;
my $f = 0;
$f = $f + $a->[$x][$y] / 2;
my $q = 7 / 2;
$i = $i + $n;
$out->[$x][$y] = $i + $f + $q;
K
);
my $c = $runner->c_source;
like $c, qr/nvptx_kernel\)\) void percuda_kernel\(float \*v_a, int \*v_out, int \*p_n, int \*percuda_dims\)/, 'signature';
like $c, qr/\bint l\d+_i = 0;/, 'int local stays int';
like $c, qr/\bfloat l\d+_f = /, 'local assigned a float becomes float';
like $c, qr/\bfloat l\d+_q = /, 'division yields float';
like $c, qr/v_a\[\(percuda_x\) \* d_a_1 \+ \(percuda_y\)\]/, 'row-major flattening';
like $c, qr/if \(percuda_x >= percuda_size_x \|\| percuda_y >= percuda_size_y\) return;/, 'bounds guard';

done_testing;
