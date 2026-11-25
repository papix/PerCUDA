use v5.40;

# Hide PDL even where it is installed, to check that it stays optional.
BEGIN {
    unshift @INC, sub ($hook, $file) {
        die "Can't locate $file (hidden by the test)\n" if $file =~ m{\APDL(?:/|\.pm\z)};
        return;
    };
}

use Test2::V0;
use lib 't/lib';
use PerCUDATest;
use PerCUDA;

ok !grep({ m{\APDL} } keys %INC), 'loading PerCUDA does not load PDL';

for my $engine (both_engines()) {
    my $r = PerCUDA->new(
        engine   => $engine,
        dim      => 2,
        argument => [ in => 'f', out => 'f' ],
        retval   => ['out'],
        source   => '$out->[$x][$y] = $in->[$x][$y] * 2;',
    );
    is scalar $r->run([ [ 1, 2, 3 ], [ 4, 5, 6 ] ], []), [ [ 2, 4, 6 ], [ 8, 10, 12 ] ], "$engine: nested arrays work";
}
ok !grep({ m{\APDL} } keys %INC), 'running kernels does not load PDL';

done_testing;
