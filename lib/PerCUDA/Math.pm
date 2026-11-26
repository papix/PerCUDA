package PerCUDA::Math;
use v5.40;
use Carp  ();
use POSIX ();

# PerCUDA's own exp, log, sin, cos, atan2 and pow for mode => 'exact'.
#
# CUDA's math functions are fast but not reproducible outside the GPU (expf
# uses the hardware ex2.approx instruction), so the exact mode computes them
# with + - * /, floor, fabs, ldexp and frexp only, which IEEE 754 and C define
# exactly. Each function is written once below as a small syntax tree and
# rendered both as CUDA C++ and as Perl that rounds every operation like the
# GPU, so the two engines give bit-identical results.
#
# Accuracy is a few ulp (t/12_math.t measures it); arguments of sin and cos
# beyond about 1e5 (single) or 1e9 (double) lose precision, identically on
# both engines.

our @FUNCTIONS = qw(exp log sin cos atan2 pow);

# ---- constants ----

sub _f32 ($v) { return unpack 'f', pack 'f', $v }

sub _factorial ($n) { my $f = 1; $f *= $_ for 2 .. $n; return $f }

my $PI = 3.14159265358979311600;

my %CONST = (
    f => {
        INV_LN2  =>  1.44269504088896341,
        LN2_HI   =>  0.693145751953125,
        LN2_LO   => 1.4286068203094172e-06,
        EXP_HI   =>  88.72283935546875,
        EXP_LO   => -103.97207641601562,
        INV_PIO2 =>  0.63661977236758138,

        # pi/2 in parts of 11 bits, so that n * PIO2_k is exact up to n = 2^13
        PIO2_1 => 1.5703125,
        PIO2_2 => 0.00048375129699707031,
        PIO2_3 => 7.5495336204767227e-08,
        PIO2_4 => 2.5633441515945189e-12,
    },
    d => {
        INV_LN2  => 1.44269504088896338700e+00,
        LN2_HI   => 6.93147180369123816490e-01,
        LN2_LO   => 1.90821492927058770002e-10,
        EXP_HI   =>  709.782712893383973096,
        EXP_LO   => -745.133219101941108420,
        INV_PIO2 =>  0.63661977236758138,

        # pi/2 in parts of 26 bits, so that n * PIO2_k is exact up to n = 2^27
        PIO2_1 => 1.5707963109016418,
        PIO2_2 => 1.5893254712295857e-08,
        PIO2_3 => 6.1232339320535943e-17,
        PIO2_4 => 6.3683171635109499e-25,
    },
);
$CONST{f}{SPLIT} = 4097;    # 2^12 + 1: splits a float into two 12-bit halves
$CONST{d}{SPLIT} = 134217729;    # 2^27 + 1
for my $t (qw(f d)) {
    $CONST{$t}{SQRT_HALF} = 0.70710678118654752440;
    $CONST{$t}{TAN_PI_12} = 0.26794919243112270;
    $CONST{$t}{SQRT3}     = 1.73205080756887719;
    $CONST{$t}{PI_6}      = $PI / 6;
    $CONST{$t}{PI_2}      = $PI / 2;
    $CONST{$t}{PI}        = $PI;
}

# Taylor coefficients; the number of terms suits each precision.
my %TERMS = (
    f => { exp => 7,  log => 5,  sin => 4, cos => 5, atan => 7 },
    d => { exp => 13, log => 11, sin => 8, cos => 9, atan => 15 }
);
for my $t (qw(f d)) {
    my $n = $TERMS{$t};
    $CONST{$t}{"E$_"} = 1 / _factorial($_)                for 0 .. $n->{exp};    # exp(r) = sum r^k / k!
    $CONST{$t}{"L$_"} = 1 / (2 * $_ + 1)                  for 0 .. $n->{log} - 1;    # log: 2 s sum z^k / (2k+1)
    $CONST{$t}{"S$_"} = (-1)**$_ / _factorial(2 * $_ + 1) for 1 .. $n->{sin};
    $CONST{$t}{"C$_"} = (-1)**$_ / _factorial(2 * $_)     for 1 .. $n->{cos};
    $CONST{$t}{"A$_"} = (-1)**$_ / (2 * $_ + 1)           for 0 .. $n->{atan} - 1;
}
$CONST{f}{$_} = _f32($CONST{f}{$_}) for keys %{ $CONST{f} };

# ---- the functions as syntax trees ----
#
# Expressions: number constants by name, variables by name, [op, a, b] for
# + - * / (rounded), comparisons < <= > >= == != and && ||, and calls
# [floor => x], [fabs => x], [ldexp => x, n], [isnan => x], [neg_zero => x].
# Statements: [set => var, expr], [frexp => mvar, evar, x], [if => cond,
# [then], [else]], [ret => expr].

sub _horner ($var, $t, $prefix, $from, $to) {
    my $p = "$prefix$to";
    $p = [ '+', [ '*', $p, $var ], "$prefix$_" ] for reverse $from .. $to - 1;
    return $p;
}

sub _functions ($t) {
    my $n = $TERMS{$t};
    my %f;
    $f{exp} = [
        [ if  => [ isnan => 'x' ],       [ [ ret => 'x' ] ] ],
        [ if  => [ '>', 'x', 'EXP_HI' ], [ [ ret => 'INF' ] ] ],
        [ if  => [ '<', 'x', 'EXP_LO' ], [ [ ret => 'ZERO' ] ] ],
        [ set => 'n',                    [ floor => [ '+', [ '*', 'x', 'INV_LN2' ], 'HALF' ] ] ],
        [ set => 'r',                    [ '-', 'x', [ '*', 'n', 'LN2_HI' ] ] ],
        [ set => 'r',                    [ '-', 'r', [ '*', 'n', 'LN2_LO' ] ] ],
        [ set => 'p',                    _horner('r', $t, 'E', 0, $n->{exp}) ],
        [ ret => [ ldexp => 'p', 'n' ] ],
    ];
    $f{log} = [
        [ if    => [ '||', [ isnan => 'x' ], [ '<', 'x', 'ZERO' ] ], [ [ ret => 'NAN' ] ] ],
        [ if    => [ '==', 'x',              'ZERO' ],               [ [ ret => 'NINF' ] ] ],
        [ if    => [ '==', 'x',              'INF' ],                [ [ ret => 'INF' ] ] ],
        [ frexp => 'm',                    'e', 'x' ],
        [ if => [ '<', 'm', 'SQRT_HALF' ], [ [ set => 'm', [ '+', 'm', 'm' ] ], [ set => 'e', [ '-', 'e', 'ONE' ] ] ] ],
        [ set => 'f', [ '-', 'm', 'ONE' ] ],
        [ set => 's', [ '/', 'f', [ '+', 'TWO', 'f' ] ] ],
        [ set => 'z', [ '*', 's', 's' ] ],
        [ set => 'p', _horner('z', $t, 'L', 0, $n->{log} - 1) ],
        [ set => 'l', [ '*', [ '+', 's', 's' ], 'p' ] ],
        [ ret => [ '+', [ '*', 'e', 'LN2_HI' ], [ '+', 'l', [ '*', 'e', 'LN2_LO' ] ] ] ],
    ];
    my @reduce = (
        [ set => 'n', [ floor => [ '+', [ '*', 'x', 'INV_PIO2' ], 'HALF' ] ] ],
        [ set => 'r', [ '-', 'x',   [ '*', 'n',    'PIO2_1' ] ] ],
        [ set => 'r', [ '-', 'r',   [ '*', 'n',    'PIO2_2' ] ] ],
        [ set => 'r', [ '-', 'r',   [ '*', 'n',    'PIO2_3' ] ] ],
        [ set => 'r', [ '-', 'r',   [ '*', 'n',    'PIO2_4' ] ] ],
        [ set => 'q', [ '-', 'n',   [ '*', 'FOUR', [ floor => [ '*', 'n', 'QUARTER' ] ] ] ] ],
        [ set => 'z', [ '*', 'r',   'r' ] ],
        [ set => 's', [ '+', 'r',   [ '*', [ '*', 'r', 'z' ], _horner('z', $t, 'S', 1, $n->{sin}) ] ] ],
        [ set => 'c', [ '+', 'ONE', [ '*', 'z',               _horner('z', $t, 'C', 1, $n->{cos}) ] ] ],
    );
    my $finite = [ if => [ '||', [ isnan => 'x' ], [ '==', [ fabs => 'x' ], 'INF' ] ], [ [ ret => 'NAN' ] ] ];
    my sub quadrant (@by_q) {
        return [
            if => [ '==', 'q', 'ZERO' ],
            [ [ ret => $by_q[0] ] ],
            [
                [
                    if => [ '==', 'q', 'ONE' ],
                    [ [ ret => $by_q[1] ] ],
                    [ [ if  => [ '==', 'q', 'TWO' ], [ [ ret => $by_q[2] ] ], [ [ ret => $by_q[3] ] ] ] ]
                ]
            ]
        ];
    }
    $f{sin} = [ $finite, @reduce, quadrant('s', 'c', [ '-', 'ZERO', 's' ], [ '-', 'ZERO', 'c' ]) ];
    $f{cos} = [ $finite, @reduce, quadrant('c', [ '-', 'ZERO', 's' ], [ '-', 'ZERO', 'c' ], 's') ];
    my sub signed ($e) { return [ if_x_negative => 'y', [ '-', 'ZERO', $e ], $e ] }    # copysign(e, y)
    $f{atan2} = [
        [ if  => [ '||', [ isnan => 'y' ], [ isnan => 'x' ] ], [ [ ret => 'NAN' ] ] ],
        [ set => 'ay', [ fabs => 'y' ] ],
        [ set => 'ax', [ fabs => 'x' ] ],
        [
            if => [ '&&', [ '==', 'ay', 'INF' ], [ '==', 'ax', 'INF' ] ],
            [
                [
                    ret => signed(
                        [ if_x_negative => 'x', [ '*', 'THREE', [ '/', 'PI', 'FOUR' ] ], [ '/', 'PI', 'FOUR' ] ]
                    )
                ]
            ]
        ],
        [ if => [ '==', 'ay', 'INF' ], [ [ ret => signed('PI_2') ] ] ],
        [
            if => [ '||', [ '==', 'ax', 'INF' ], [ '==', 'ay', 'ZERO' ] ],
            [ [ ret => signed([ if_x_negative => 'x', 'PI', 'ZERO' ]) ] ]
        ],
        [ if  => [ '==', 'ax', 'ZERO' ], [ [ ret => signed('PI_2') ] ] ],
        [ set => 'w', [ '>', 'ay', 'ax' ] ],
        [ set => 't', [ select => 'w', [ '/', 'ax', 'ay' ], [ '/', 'ay', 'ax' ] ] ],
        [ set => 'o', 'ZERO' ],
        [
            if => [ '>', 't', 'TAN_PI_12' ],
            [
                [ set => 't', [ '/', [ '-', [ '*', 't', 'SQRT3' ], 'ONE' ], [ '+', 't', 'SQRT3' ] ] ],
                [ set => 'o', 'PI_6' ],
            ]
        ],
        [ set => 'z',                  [ '*', 't', 't' ] ],
        [ set => 'a',                  [ '+', 'o', [ '*', 't', _horner('z', $t, 'A', 0, $n->{atan} - 1) ] ] ],
        [ if  => 'w',                  [ [ set => 'a', [ '-', 'PI_2', 'a' ] ] ] ],
        [ if  => [ '<', 'x', 'ZERO' ], [ [ set => 'a', [ '-', 'PI',   'a' ] ] ] ],
        [ ret => signed('a') ],
    ];

    # pow(x, y) = exp(y log x). The error of log x is multiplied by y log x, so
    # log x and the product are carried as hi + lo pairs (Dekker's exact
    # products, which need no fused multiply-add) and exp(hi + lo) is
    # exp(hi) (1 + lo).
    my sub split_into ($v, $hi, $lo) {    # $v = $hi + $lo with half-width $hi
        return (
            [ set => "${hi}c", [ '*', 'SPLIT',  $v ] ],
            [ set => $hi,      [ '-', "${hi}c", [ '-', "${hi}c", $v ] ] ],
            [ set => $lo,      [ '-', $v,       $hi ] ],
        );
    }
    my sub two_prod ($a, $b, $p, $err) {    # $p + $err = $a * $b exactly
        return (
            [ set => $p, [ '*', $a, $b ] ],
            split_into($a, 'ah', 'al'),
            split_into($b, 'bh', 'bl'),
            [
                set => $err,
                [
                    '+',
                    [ '+', [ '+', [ '-', [ '*', 'ah', 'bh' ], $p ], [ '*', 'ah', 'bl' ] ], [ '*', 'al', 'bh' ] ],
                    [ '*', 'al',                                                           'bl' ]
                ]
            ],
        );
    }
    my sub two_sum ($a, $b, $s, $err) {    # $s + $err = $a + $b exactly
        return (
            [ set => $s,   [ '+', $a,                             $b ] ],
            [ set => 'bv', [ '-', $s,                             $a ] ],
            [ set => $err, [ '+', [ '-', $a, [ '-', $s, 'bv' ] ], [ '-', $b, 'bv' ] ] ],
        );
    }
    $f{pow} = [
        [ if  => [ '||', [ '==', 'y', 'ZERO' ], [ '==', 'x', 'ONE' ] ], [ [ ret => 'ONE' ] ] ],
        [ if  => [ '||', [ isnan => 'x' ],      [ isnan => 'y' ] ],     [ [ ret => 'NAN' ] ] ],
        [ if  => [ '==', 'x', 'ZERO' ], [ [ ret => [ select => [ '>', 'y', 'ZERO' ], 'ZERO', 'INF' ] ] ] ],
        [ if  => [ '&&', [ '<', 'x', 'ZERO' ], [ '!=', [ floor => 'y' ], 'y' ] ], [ [ ret => 'NAN' ] ] ],
        [ set => 'ax', [ fabs => 'x' ] ],
        [
            if => [ '==', 'ax', 'INF' ],
            [ [ set => 'r', [ select => [ '>', 'y', 'ZERO' ], 'INF', 'ZERO' ] ] ],
            [
                # log(ax) = e ln 2 + 2 atanh(s), s = f / (2 + f), f = m - 1
                [ frexp => 'm', 'e', 'ax' ],
                [
                    if => [ '<', 'm', 'SQRT_HALF' ],
                    [ [ set => 'm', [ '+', 'm', 'm' ] ], [ set => 'e', [ '-', 'e', 'ONE' ] ] ]
                ],
                [ set => 'f',  [ '-', 'm',   'ONE' ] ],
                [ set => 'dh', [ '+', 'TWO', 'f' ] ],
                [ set => 'dl', [ '-', 'f',   [ '-', 'dh', 'TWO' ] ] ],
                [ set => 'sh', [ '/', 'f',   'dh' ] ],
                two_prod('sh', 'dh', 'p', 'pe'),
                [ set => 'sl', [ '/', [ '-', [ '-', [ '-', 'f', 'p' ], 'pe' ], [ '*', 'sh', 'dl' ] ], 'dh' ] ],
                [ set => 'z',  [ '*', 'sh',                                                           'sh' ] ],
                [
                    set => 'lt',
                    [
                        '+',
                        [ '*', [ '*', [ '+', 'sh', 'sh' ], 'z' ], _horner('z', $t, 'L', 1, $n->{log} - 1) ],
                        [ '+', 'sl',                              'sl' ]
                    ]
                ],
                two_sum([ '*', 'e', 'LN2_HI' ], [ '+', 'sh', 'sh' ], 'lh', 'll'),
                [ set => 'll', [ '+', 'll', [ '+', 'lt', [ '*', 'e', 'LN2_LO' ] ] ] ],
                [ set => 'h',  [ '+', 'lh', 'll' ] ],
                [ set => 'll', [ '-', 'll', [ '-', 'h', 'lh' ] ] ],

                # t = y log(ax) as th + tl, then exp(th) (1 + tl)
                [ set => 'th', [ '*', 'y', 'h' ] ],
                [
                    if => [ '>', 'th', [ '+', 'EXP_HI', 'ONE' ] ],
                    [ [ set => 'r', 'INF' ] ],
                    [
                        [
                            if => [ '<', 'th', [ '-', 'EXP_LO', 'ONE' ] ],
                            [ [ set => 'r', 'ZERO' ] ],
                            [
                                two_prod('y', 'h', 'th', 'te'),
                                [ set => 'tl', [ '+', 'te', [ '*', 'y', 'll' ] ] ],
                                [ set => 'r',  [ call => 'exp', 'th' ] ],
                                [ set => 'r',  [ '+', 'r', [ '*', 'r', 'tl' ] ] ],
                            ]
                        ],
                    ]
                ],
            ]
        ],
        [
            if => [ '<', 'x', 'ZERO' ],
            [
                [ if  => [ '!=', [ floor => 'y' ], 'y' ], [ [ ret => 'NAN' ] ] ],
                [ set => 'h',                             [ '*', 'y', 'HALF' ] ],
                [ if  => [ '!=', [ floor => 'h' ], 'h' ], [ [ set => 'r', [ '-', 'ZERO', 'r' ] ] ] ],
            ]
        ],
        [ ret => 'r' ],
    ];
    return \%f;
}

my %ARGS    = (exp  => ['x'], log => ['x'], sin => ['x'], cos => ['x'], atan2 => [ 'y', 'x' ], pow => [ 'x', 'y' ]);
my %LITERAL = (ZERO => 0, HALF => 0.5, QUARTER => 0.25, ONE => 1, TWO => 2, THREE => 3, FOUR => 4);

# ---- rendering ----

sub _vars ($stmts, $seen = {}) {
    for my $s (@$stmts) {
        next unless ref $s eq 'ARRAY';
        $seen->{ $s->[1] } = 1 if $s->[0] eq 'set';
        @$seen{ $s->[1], $s->[2] } = (1, 1) if $s->[0] eq 'frexp';
        if ($s->[0] eq 'if') { _vars($s->[2], $seen); _vars($s->[3] // [], $seen) }
    }
    return $seen;
}

sub _c_const ($t, $name) {
    return $t eq 'f' ? '__int_as_float(0x7f800000)'    : '__longlong_as_double(0x7ff0000000000000LL)' if $name eq 'INF';
    return $t eq 'f' ? '(-__int_as_float(0x7f800000))' : '(-__longlong_as_double(0x7ff0000000000000LL))'
        if $name eq 'NINF';
    return $t eq 'f' ? '__int_as_float(0x7fffffff)' : '__longlong_as_double(0x7fffffffffffffffLL)' if $name eq 'NAN';
    my $v = $LITERAL{$name} // $CONST{$t}{$name} // Carp::confess("unknown constant $name");
    my $s = sprintf($t eq 'f' ? '%.9g' : '%.17g', $v);
    $s .= '.0' unless $s =~ /[.e]/;
    return $t eq 'f' ? "${s}f" : $s;
}

sub _c_expr ($t, $e) {
    my $fn = $t eq 'f' ? sub ($name) { "${name}f" } : sub ($name) { $name };
    return $e =~ /\A[A-Z]/ ? _c_const($t, $e) : $e unless ref $e;
    my ($op, @a) = @$e;
    return '(' . _c_expr($t, $a[0]) . " $op " . _c_expr($t, $a[1]) . ')' if $op =~ /\A(?:[-+*\/<>]|[<>=!]=|&&|\|\|)\z/;
    return $fn->('floor') . '(' . _c_expr($t, $a[0]) . ')'               if $op eq 'floor';
    return $fn->('fabs') . '(' . _c_expr($t, $a[0]) . ')'                if $op eq 'fabs';
    return $fn->('ldexp') . '(' . _c_expr($t, $a[0]) . ', (int)' . _c_expr($t, $a[1]) . ')' if $op eq 'ldexp';
    return '(' . _c_expr($t, $a[0]) . ' != ' . _c_expr($t, $a[0]) . ')'                     if $op eq 'isnan';
    return '(signbit(' . _c_expr($t, $a[0]) . ') ? ' . _c_expr($t, $a[1]) . ' : ' . _c_expr($t, $a[2]) . ')'
        if $op eq 'if_x_negative';
    return '(' . _c_expr($t, $a[0]) . ' ? ' . _c_expr($t, $a[1]) . ' : ' . _c_expr($t, $a[2]) . ')' if $op eq 'select';
    return "percuda_$a[0]_$t(" . _c_expr($t, $a[1]) . ')'                                           if $op eq 'call';
    Carp::confess("unknown operation $op");
}

sub _c_stmts ($t, $stmts, $ind) {
    my $out = '';
    for my $s (@$stmts) {
        my ($op, @a) = @$s;
        if    ($op eq 'set') { $out .= "$ind$a[0] = " . _c_expr($t, $a[1]) . ";\n" }
        elsif ($op eq 'ret') { $out .= "${ind}return " . _c_expr($t, $a[0]) . ";\n" }
        elsif ($op eq 'frexp') {
            my $fn = $t eq 'f' ? 'frexpf' : 'frexp';
            $out .= "$ind\{ int percuda_e; $a[0] = $fn($a[2], &percuda_e); $a[1] = percuda_e; }\n";
        }
        elsif ($op eq 'if') {
            $out .= "${ind}if (" . _c_expr($t, $a[0]) . ") {\n" . _c_stmts($t, $a[1], "$ind    ") . "$ind}";
            $out .= " else {\n" . _c_stmts($t, $a[2], "$ind    ") . "$ind}" if $a[2];
            $out .= "\n";
        }
        else { Carp::confess("unknown statement $op") }
    }
    return $out;
}

# CUDA C++ definitions of percuda_<name>_f and percuda_<name>_d.
sub c_prelude () {
    my $out = '';
    for my $t (qw(f d)) {
        my $ctype = $t eq 'f' ? 'float' : 'double';
        my $f     = _functions($t);
        for my $name (@FUNCTIONS) {
            my @params = map { "$ctype $_" } @{ $ARGS{$name} };
            my %vars   = %{ _vars($f->{$name}) };
            delete @vars{ @{ $ARGS{$name} } };
            my $decl = %vars ? "    $ctype " . join(', ', sort keys %vars) . ";\n" : '';
            $out .=
                  "static __device__ $ctype percuda_${name}_$t("
                . join(', ', @params)
                . ") {\n$decl"
                . _c_stmts($t, $f->{$name}, '    ') . "}\n";
        }
    }
    return $out;
}

sub _pl_const ($t, $name) {
    return '$PerCUDA::Math::INF'    if $name eq 'INF';
    return '(-$PerCUDA::Math::INF)' if $name eq 'NINF';
    return '$PerCUDA::Math::NAN'    if $name eq 'NAN';
    my $v = $LITERAL{$name} // $CONST{$t}{$name} // Carp::confess("unknown constant $name");
    my $s = sprintf('%.17g', $v);
    return $s =~ /[.e]/ ? $s : "$s.0";
}

sub _pl_expr ($t, $e) {
    my $round = $t eq 'f' ? sub ($code) { "unpack('f', pack('f', $code))" } : sub ($code) { "($code)" };
    return $e =~ /\A[A-Z]/ ? _pl_const($t, $e) : "\$$e" unless ref $e;
    my ($op, @a) = @$e;
    return $round->(_pl_expr($t, $a[0]) . " $op " . _pl_expr($t, $a[1]))   if $op =~ /\A[-+*\/]\z/;
    return '(' . _pl_expr($t, $a[0]) . " $op " . _pl_expr($t, $a[1]) . ')' if $op =~ /\A(?:[<>]|[<>=!]=|&&|\|\|)\z/;
    return 'POSIX::floor(' . _pl_expr($t, $a[0]) . ')'                                        if $op eq 'floor';
    return 'abs(' . _pl_expr($t, $a[0]) . ')'                                                 if $op eq 'fabs';
    return $round->('POSIX::ldexp(' . _pl_expr($t, $a[0]) . ', ' . _pl_expr($t, $a[1]) . ')') if $op eq 'ldexp';
    return '(' . _pl_expr($t, $a[0]) . ' != ' . _pl_expr($t, $a[0]) . ')'                     if $op eq 'isnan';
    return
          '(PerCUDA::Math::signbit('
        . _pl_expr($t, $a[0]) . ') ? '
        . _pl_expr($t, $a[1]) . ' : '
        . _pl_expr($t, $a[2]) . ')'
        if $op eq 'if_x_negative';
    return '(' . _pl_expr($t, $a[0]) . ' ? ' . _pl_expr($t, $a[1]) . ' : ' . _pl_expr($t, $a[2]) . ')'
        if $op eq 'select';
    return "PerCUDA::Math::$a[0]_$t(" . _pl_expr($t, $a[1]) . ')' if $op eq 'call';
    Carp::confess("unknown operation $op");
}

sub _pl_stmts ($t, $stmts, $ind) {
    my $out = '';
    for my $s (@$stmts) {
        my ($op, @a) = @$s;
        if    ($op eq 'set')   { $out .= "$ind\$$a[0] = " . _pl_expr($t, $a[1]) . ";\n" }
        elsif ($op eq 'ret')   { $out .= "${ind}return " . _pl_expr($t, $a[0]) . ";\n" }
        elsif ($op eq 'frexp') { $out .= "$ind(\$$a[0], \$$a[1]) = POSIX::frexp(\$$a[2]);\n" }
        elsif ($op eq 'if') {
            $out .= "${ind}if (" . _pl_expr($t, $a[0]) . ") {\n" . _pl_stmts($t, $a[1], "$ind    ") . "$ind}";
            $out .= " else {\n" . _pl_stmts($t, $a[2], "$ind    ") . "$ind}" if $a[2];
            $out .= "\n";
        }
        else { Carp::confess("unknown statement $op") }
    }
    return $out;
}

our $INF = 9**9**9;
our $NAN = -sin($INF);

sub signbit ($v) { return unpack('Q', pack('d', $v)) >> 63 }

sub perl_source () {
    my $out = "package PerCUDA::Math;\nno warnings qw(uninitialized numeric);\n";
    for my $t (qw(f d)) {
        my $f = _functions($t);
        for my $name (@FUNCTIONS) {
            my @args = @{ $ARGS{$name} };
            my %vars = %{ _vars($f->{$name}) };
            delete @vars{@args};
            $out .= "sub ${name}_$t {\n    my (" . join(', ', map { "\$$_" } @args) . ") = \@_;\n";
            $out .= '    my (' . join(', ', map { "\$$_" } sort keys %vars) . ");\n" if %vars;
            $out .= _pl_stmts($t, $f->{$name}, '    ') . "}\n";
        }
    }
    return $out . "1;\n";
}

eval perl_source() or Carp::croak("PerCUDA: cannot build the math functions: $@");    ## no critic (ProhibitStringyEval)

1;
