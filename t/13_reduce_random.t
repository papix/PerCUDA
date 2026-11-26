use v5.40;
use Test2::V0 '!float';
use lib 't/lib';
use PerCUDATest;
use PerCUDA;

my sub runner ($engine, $type, %args) {
    return PerCUDA->new(
        engine   => $engine,
        argument => [ a => $type ],
        source   => 'my $x = thread_index(); $a->[$x] = 1;',
        %args
    );
}
my sub same ($p, $g) { return defined $p && defined $g ? ("$p" eq "$g" || $p == $g) : !defined $p && !defined $g }

srand 11;
my %data = (
    f => [ map { (rand(2) - 1) * 10**(rand(6) - 3) } 1 .. 300_000 ]
    ,    # more than 256 x 1024: threads fold several elements
    d => [ map { (rand(2) - 1) * 10**(rand(6) - 3) } 1 .. 50_000 ],
    i => [ map { int(rand(4e9) - 2e9) } 1 .. 50_000 ],    # sums beyond 32 bits
);

subtest 'reductions agree bit for bit between the engines' => sub {
    for my $type (sort keys %data) {
        my %result;
        for my $engine (both_engines()) {
            my $a = runner($engine, $type)->array($data{$type}, type => $type);
            $result{$engine} = [ $a->sum, $a->min, $a->max, $a->argmin->[0], $a->argmax->[0] ];
        }
        my $perl = $result{perl};
        my ($min_i) = sort { $data{$type}[$a] <=> $data{$type}[$b] || $a <=> $b } 0 .. $#{ $data{$type} };
        is $perl->[3], $min_i, "$type: argmin is the first minimum";
        if ($type eq 'i') {
            my $sum = 0;
            $sum += $_ for @{ $data{i} };
            is $perl->[0], $sum, 'int sum is exact (64-bit)';
        }
        else {
            my $sum = 0;
            $sum += $_ for @{ $data{$type} };
            ok abs($perl->[0] - $sum) <= 1e-3 * abs($sum) + 1e-2, "$type: sum is close to the exact sum";
        }
        next unless $result{gpu};
        ok same($result{gpu}[$_], $perl->[$_]), "$type: " . (qw(sum min max argmin argmax))[$_] . ' identical'
            for 0 .. 4;
    }
};

subtest 'reductions in every mode and on 2D arrays' => sub {
    my $grid = [ [ 3, 1, 4 ], [ 1, 5, 9 ], [ 2, 6, 5 ] ];
    for my $engine (both_engines()) {
        for my $mode (qw(fast compat exact)) {
            my $a = runner($engine, 'i', mode => $mode)->array($grid, type => 'i');
            is [ $a->sum, $a->min, $a->max ], [ 36, 1, 9 ], "$engine $mode: sum, min, max";
            is $a->argmin, [ 0, 1 ], "$engine $mode: argmin is the first minimum, outermost first";
            is $a->argmax, [ 1, 2 ], "$engine $mode: argmax";
        }
    }
};

subtest 'nan, empty arrays' => sub {
    my $nan = 9**9**9 / 9**9**9;
    for my $engine (both_engines()) {
        my $r = runner($engine, 'd', precision => 'double');
        my $a = $r->array([ $nan, 2, $nan, -1, 7 ], type => 'd');
        is [ $a->min, $a->max ],                             [ -1, 7 ], "$engine: min and max ignore nan";
        is $a->argmin,                                       [3],       "$engine: argmin skips nan";
        is "" . $r->array([ $nan, $nan ], type => 'd')->min, 'NaN',     "$engine: all nan gives nan";
        my $empty = $r->array(type => 'd', dims => [0]);
        is $empty->sum,    0,     "$engine: sum of nothing is 0";
        is $empty->min,    undef, "$engine: min of nothing is undef";
        is $empty->argmax, undef, "$engine: argmax of nothing is undef";
    }
};

subtest 'random' => sub {
    my $kernel = 'my ($x, $y) = thread_index(); $out->[$x][$y] = random($seed, $x, $y);';
    my %result;
    for my $engine (both_engines()) {
        for my $precision (qw(single double)) {
            for my $mode (qw(fast compat exact)) {
                my $r = PerCUDA->new(
                    engine    => $engine,
                    mode      => $mode,
                    precision => $precision,
                    argument  => [ out => $precision eq 'single' ? 'f' : 'd', seed => 'i' ],
                    retval    => ['out'],
                    source    => $kernel,
                );
                $result{$engine}{$precision}{$mode} =
                    [ map { @$_ } @{ $r->run([ map { [ (0) x 200 ] } 1 .. 200 ], 42) } ];
            }
        }
    }
    for my $precision (qw(single double)) {
        my $ref = $result{perl}{$precision}{compat};
        for my $engine (sort keys %result) {
            for my $mode (qw(fast compat exact)) {
                is $result{$engine}{$precision}{$mode}, $ref, "$precision: $engine $mode gives the same numbers";
            }
        }
        my $n = @$ref;
        my ($mean, $var) = (0, 0);
        $mean += $_ / $n              for @$ref;
        $var  += ($_ - $mean)**2 / $n for @$ref;
        ok abs($mean - 0.5) < 0.01,            "$precision: mean $mean";
        ok abs($var - 1 / 12) < 0.003,         "$precision: variance $var";
        ok !grep({ $_ < 0 || $_ >= 1 } @$ref), "$precision: in [0, 1)";
        my @bins = (0) x 16;
        $bins[ int($_ * 16) ]++ for @$ref;
        my $chi2 = 0;
        $chi2 += ($_ - $n / 16)**2 / ($n / 16) for @bins;
        ok $chi2 < 40, "$precision: chi-square over 16 bins $chi2 (15 degrees of freedom)";
        my $corr = 0;
        $corr += ($ref->[$_] - 0.5) * ($ref->[ $_ + 1 ] - 0.5) / ($n - 1) / (1 / 12) for 0 .. $n - 2;
        ok abs($corr) < 0.02, "$precision: neighbours are uncorrelated ($corr)";
    }
    like dies {
        PerCUDA->new(
            engine   => 'perl',
            argument => [ o => 'f' ],
            source   => 'my $x = thread_index(); $o->[$x] = random();'
        )
    }, qr/random takes 1 to 4 arguments/, 'random needs a seed';
};

done_testing;
