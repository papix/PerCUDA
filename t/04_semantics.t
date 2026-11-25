use v5.40;
use Test::More;
use lib 't/lib';
use PerCUDATest;
use PerCUDA;

plan skip_all => 'no GPU' unless gpu_available();

# Each kernel runs on both engines and the results must agree.
my @cases = (
    {
        name => 'int arithmetic, division, modulus with negatives',
        argument => [ in => 'i', out => 'f' ],
        source => <<'K',
my $v = $in->[$x];
my $r = 0;
$r += $v % 7;
$r += $v % -4;
$r += -$v / 4;
$r += int($v / 3);
$r += $v * 3 - 2;
$out->[$x] = $r;
K
    },
    {
        name => 'bit operations and shifts',
        argument => [ in => 'i', out => 'i' ],
        source => <<'K',
my $v = $in->[$x] + 60;    # bit operations on negative numbers differ (perl uses unsigned 64-bit)
$out->[$x] = (($v & 5) | ($v ^ 3)) + ($v << 2) + ($v >> 1) + (~$v & 255);
K
    },
    {
        name => 'control flow: last, next, while, foreach, ternary, return',
        argument => [ in => 'i', out => 'i' ],
        source => <<'K',
my $v = $in->[$x];
my $acc = 0;
for (my $i = 0; $i < 50; $i++) {
    next if $i % 3 == 0;
    last if $i > $v + 10;
    $acc += $i;
}
for my $j (-2 .. 4) {
    $j = 100;    # does not affect the iteration
    $acc += $j;
}
my $k = $v;
while ($k > 1) { $k = $k % 2 ? 3 * $k + 1 : $k / 2; $acc++; }
unless ($v % 5) { $out->[$x] = -1; return; }
$out->[$x] = $acc > 200 ? $acc - 200 : $acc;
K
    },
    {
        name => 'logical operators return the deciding operand',
        argument => [ in => 'i', out => 'i' ],
        source => <<'K',
my $v = $in->[$x] % 3;
my $a = $v || 7;
my $b = $v && 9;
my $c = 0;
$c ||= 4;
$c &&= $c + $v;
$out->[$x] = $a * 100 + $b * 10 + $c + (!$v) + (not $v and 1);
K
    },
    {
        name => 'power, sqrt, abs',
        argument => [ in => 'i', out => 'f' ],
        source => <<'K',
my $v = $in->[$x] - 10;
$out->[$x] = $v ** 3 + 2 ** 10 + abs($v) + sqrt abs $v + 1 + (0.5 * $v) ** 2 + 2 ** -2;
K
    },
    {
        name => 'double precision',
        argument => [ in => 'd', out => 'd' ],
        source => <<'K',
my $s = 0;
for my $i (1 .. 100) { $s += $in->[$x] / $i; }
$out->[$x] = $s;
K
    },
);

my $N = 300;
for my $case (@cases) {
    subtest $case->{name} => sub {
        my %result;
        for my $engine (qw(perl gpu)) {
            my $r = PerCUDA->new(
                dim      => 1,
                engine   => $engine,
                argument => $case->{argument},
                retval   => ['out'],
                source   => $case->{source},
            );
            $result{$engine} = $r->run([ map { $_ - 50 } 0 .. $N - 1 ], []);
        }
        my @bad = grep {
            my ($p, $g) = ($result{perl}[$_], $result{gpu}[$_]);
            abs($p - $g) > 1e-4 * (1 + abs $p);
        } 0 .. $N - 1;
        is scalar @bad, 0, 'perl and gpu agree'
            or diag join "\n", map { "[$_] perl=$result{perl}[$_] gpu=$result{gpu}[$_]" } @bad[ 0 .. ($#bad < 4 ? $#bad : 4) ];
    };
}

subtest '3D grid and scalar arguments' => sub {
    my %result;
    my $cube = [ map { my $i = $_; [ map { my $j = $_; [ map { $i * 100 + $j * 10 + $_ } 0 .. 9 ] } 0 .. 6 ] } 0 .. 4 ];
    for my $engine (qw(perl gpu)) {
        my $r = PerCUDA->new(
            dim      => 3,
            engine   => $engine,
            argument => [ c => 'i', out => 'f', scale => 'f' ],
            retval   => ['out'],
            source   => '$out->[$x][$y][$z] = $c->[$x][$y][$z] * $scale + $size_x * $size_y * $size_z;',
        );
        $result{$engine} = $r->run($cube, [], 0.5);
    }
    is_deeply $result{gpu}, $result{perl}, 'perl and gpu agree';
    is $result{gpu}[4][6][9], 469 * 0.5 + 350, 'value';
};

done_testing;
