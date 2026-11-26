use v5.40;
use Test2::V0;
use lib 't/lib';
use PerCUDATest;
use PerCUDA;

my $KERNEL = <<'K';
my $v = $in->[$x];
my $w = $in2->[$x];
$out->[$x][0] = exp($v / 4);
$out->[$x][1] = log($v);
$out->[$x][2] = sin($v);
$out->[$x][3] = cos($v);
$out->[$x][4] = atan2($w, $v);
$out->[$x][5] = $v ** $w;
$out->[$x][6] = floor($v) + ceil($v) * 1000;
$out->[$x][7] = sqrt($v);
$out->[$x][8] = abs($v) ** 2.5 + 2 ** $w;
K
my @NAMES = qw(exp log sin cos atan2 pow floor/ceil sqrt pow2);

srand 7;
my @in = (0, -0.0, 1, -1, 0.5, 2, 3.14159265, -2.718, 88, -103, 1e-30, 1e30, 9**9**9, -9**9**9);
push @in, map { (rand(2) - 1) * (10**(rand(8) - 3)) } 1 .. 2000;
my @in2 = map { (rand(2) - 1) * 4 } @in;
$in2[$_] = int $in2[$_] for grep { $_ % 5 == 0 } 0 .. $#in2;    # integer exponents for negative bases

my sub run ($engine, $mode, $type) {
    my $r = PerCUDA->new(
        engine   => $engine,
        mode     => $mode,
        size     => 'in',
        argument => [ in => $type, in2 => $type, out => $type ],
        retval   => ['out'],
        source   => $KERNEL,
        $type eq 'd' ? (precision => 'double') : (),
    );
    return $r->run([@in], [@in2], [ map { [ (0) x 9 ] } @in ]);
}

my sub same ($p, $g) { return "$p" eq "$g" || $p == $g }

my sub ulps ($p, $g, $type) {
    return 0       if same($p, $g);
    return 9**9**9 if $p != $p || $g != $g || abs($p) == 9**9**9 || abs($g) == 9**9**9;
    my $scale = abs($p) > abs($g) ? abs($p) : abs($g);
    my $ulp   = $scale * ($type eq 'f' ? 2**-23 : 2**-52);
    my $min   = $type eq 'f' ? 2**-149 : 2**-1074;
    return abs($p - $g) / ($ulp > $min ? $ulp : $min);
}

for my $type (qw(f d)) {
    subtest "perl engine without errors ($type)" => sub {
        for my $mode (qw(fast compat exact)) {
            my $out = run('perl', $mode, $type);
            my ($zero, $minus) = (0, 3);    # rows of 0 and -1
            is "$out->[$zero][1]",  '-Inf',   "$mode: log(0) is -inf";
            is "$out->[$minus][1]", 'NaN',    "$mode: log(-1) is nan";
            is "$out->[$minus][7]", 'NaN',    "$mode: sqrt(-1) is nan";
            is $out->[5][6],        2 + 2000, "$mode: floor and ceil";
            ok abs($out->[4][0] - exp(0.125)) < 1e-6, "$mode: exp";
        }
    };

    next unless gpu_available();

    subtest "exact: gpu and perl agree bit for bit ($type)" => sub {
        my ($gpu, $perl) = (run('gpu', 'exact', $type), run('perl', 'exact', $type));
        for my $k (0 .. $#NAMES) {
            my @bad = grep { !same($perl->[$_][$k], $gpu->[$_][$k]) } 0 .. $#in;
            is scalar @bad, 0, $NAMES[$k]
                or diag join "\n",
                map { "in=$in[$_] in2=$in2[$_]: perl=$perl->[$_][$k] gpu=$gpu->[$_][$k]" } @bad[ 0 .. 2 ];
        }
    };

    subtest "compat: gpu (CUDA math) and perl differ by a few ulp at most ($type)" => sub {
        my ($gpu, $perl) = (run('gpu', 'compat', $type), run('perl', 'compat', $type));
        for my $k (0 .. $#NAMES) {
            next if $NAMES[$k] =~ /pow/;    # the error of pow grows with the result; compared below
            my ($worst) = sort { $b <=> $a } map { ulps($perl->[$_][$k], $gpu->[$_][$k], $type) } 0 .. $#in;
            cmp_ok $worst, '<=', 4, "$NAMES[$k] (worst $worst ulp)";
        }
        my @bad = grep {
            my ($p, $g) = ($perl->[$_][5], $gpu->[$_][5]);
            !same($p, $g) && !(abs($p - $g) <= 1e-5 * abs($p))
        } 0 .. $#in;
        is scalar @bad, 0, 'pow within 1e-5 relative';
    };
}

subtest 'accuracy of the exact mode functions' => sub {

    # PerCUDA::Math against Perl's double precision functions
    my $out = run('perl', 'exact', 'd');
    my %worst;
    for my $i (0 .. $#in) {
        my ($v, $w) = ($in[$i], $in2[$i]);
        my %ref = (exp => exp($v / 4), sin => sin($v), cos => cos($v), atan2 => atan2($w, $v));
        $ref{log} = log($v) if $v > 0;
        for my $k (0 .. 4) {
            my $r = $ref{ $NAMES[$k] } // next;
            next if abs($r) < 1e-3 || abs($r) > 1e300 || ($k == 2 || $k == 3) && abs($v) > 1e8;
            my $u = ulps($out->[$i][$k], $r, 'd');
            $worst{ $NAMES[$k] } = $u if $u > ($worst{ $NAMES[$k] } // 0);
        }
    }
    cmp_ok $worst{$_}, '<=', 4, "$_ within 4 ulp (worst $worst{$_})" for sort keys %worst;
};

like dies { PerCUDA->new(mode => 'slow', argument => [ a => 'f' ], source => '$a->[$x] = 1;') }, qr/mode must be/,
    'rejects unknown modes';

done_testing;
