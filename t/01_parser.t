use v5.40;
use Test2::V0;
use PerCUDA;

sub build ($source, %args) {
    return PerCUDA->new(
        engine   => 'perl',
        argument => [ a => 'f', n => 'i' ],
        source   => $source,
        %args,
    );
}

subtest 'accepted syntax' => sub {
    ok build(<<'K'), 'rich kernel parses';
my $s = 0;   # comment
for (my $i = 0; $i < $n; $i++) { next if $i % 2; $s += $i; }
for my $j (0 .. $n - 1) { last unless $j < 3; $s -= $j; }
while ($s > 100) { $s = $s >> 1; }
until ($s < 50) { $s--; }
if ($s == 1) { $s = 2 } elsif (not $s) { $s = 3 } else { $s = $s ** 2 }
unless ($s) { return; }
$s = $s > 0 ? sqrt $s + 1 : abs(-$s);
$s ||= 1; $s &&= int($s / 3);
{ my $t = $s; $s = $t; }
$a->[$x] = $s and $a->[$x] += 0x10 + 1_000 + 1e-3 + .5;
K
};

my @errors = (
    [ '$a->[$x] = "str";',              qr/string literals are not supported/ ],
    [ '$a->[$x] = $undeclared;',        qr/\$undeclared is not declared/ ],
    [ 'my @list;',                      qr/unsupported character .@./ ],
    [ 'my %h;',                         qr/unsupported character|only scalar/ ],
    [ '$a->[$x] = sin(1);',             qr/function 'sin' is not supported/ ],
    [ 'last;',                          qr/'last' outside of a loop/ ],
    [ 'for my $i (0..3) { { last; } }', qr/'last' inside a bare block/ ],
    [ '$a->[$x] = $a->[$x][0];',        qr/used with 1 and 2 subscripts/ ],
    [ '$a->[$x] = $b->[0];',            qr/\$b is not an argument/ ],
    [ q{$a->[$x] = 1; $a = 2;},         qr/must be used with subscripts/ ],
    [ '$a->[$x] = (',                   qr/syntax error/ ],
    [ '$a->[$x] = 1 .. 2;',             qr/range operator/ ],
    [ '$a->[$x][$x][$x][$x] = 1;',      qr/deeper than 3/ ],
    [ '$a->[$x] = 2 ** 0.5;',           qr/integer exponent/ ],
);
for my $case (@errors) {
    my ($src, $re) = @$case;
    like dies { build($src) }, $re, "rejects: $src";
}

like dies { build('$a->[$x] = 1;',  argument => [ a => 'q' ]) }, qr/unknown type 'q'/, 'rejects unknown type';
like dies { build('$a->[$x] = 1;',  argument => [ x => 'f' ]) }, qr/reserved/,         'rejects reserved argument name';
like dies { build('$a->[$x] = $n;', retval   => ['n']) },        qr/must be an array/, 'retval must be an array';
like dies { build('$a->[$x] = 1;',  engine   => 'cuda') },       qr/engine must be/,   'rejects unknown engine';

done_testing;
