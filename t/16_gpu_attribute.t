use v5.40;
use Test2::V0;
use lib 't/lib';
use PerCUDATest;

my @warnings;
local $SIG{__WARN__} = sub { push @warnings, @_ };
use PerCUDA;
use PerCUDA::Parser;
use PerCUDA::Analyzer;

# Kernels as subs with :GPU; both engines run the same body.

sub add : GPU(float, float, float) ($a, $b, $out) {
    my ($x, $y) = thread_index();
    $out->[$x][$y] = $a->[$x][$y] + $b->[$x][$y];    # {braces} in a comment
}

sub add_perl : GPU(float, float, float, { engine => 'perl' }) ($a, $b, $out) {
    my ($x, $y) = thread_index();
    $out->[$x][$y] = $a->[$x][$y] + $b->[$x][$y];
}

sub squares : GPU({ size => 'n' }) ($out, $n) {
    my $x = thread_index();
    $out->[$x] = $x * $x;
}

sub scaled : GPU({ precision => 'double' }) ($in, $out, $s) {
    my $x = thread_index();
    if ($in->[$x] < 0) {
        $out->[$x] = 0.0;
        return;
    }
    $out->[$x] = $in->[$x] * $s;
}

my $p = [ [ 1,  2 ],  [ 3,  4 ] ];
my $q = [ [ 10, 20 ], [ 30, 40 ] ];

is PerCUDA->runner(\&add)->engine, (gpu_available() ? 'gpu' : 'perl'), 'the default engine';
is scalar add($p, $q, []), [ [ 11, 22 ], [ 33, 44 ] ], 'a :GPU sub runs as a kernel';
my $out = [];
add($p, $q, $out);
is $out,                                 [ [ 11, 22 ], [ 33, 44 ] ], 'the written array is filled in place';
is PerCUDA->runner(\&add_perl)->engine,  'perl',                     'options in the attribute';
is scalar add_perl($p, $q, []),          [ [ 11, 22 ], [ 33, 44 ] ], 'perl engine';
is scalar squares([], 5),                [ 0, 1, 4, 9, 16 ],         'size => n (an integer scalar) gives the grid';
is scalar scaled([ 1, -1, 3 ], [], 0.5), [ 0.5, 0, 1.5 ],            'return and double precision';
like dies {
    PerCUDA->runner(sub { })
},
    qr/not a :GPU sub/,
    'runner of an ordinary sub';

sub checked : GPU(float, { check => 1 }) ($in) {
    my $x = thread_index();
    $in->[ $x + 1 ] = 0;
}
my $line = __LINE__ - 2;
like dies { checked([ 1, 2 ]) }, qr/at kernel line $line:/, 'kernel lines are lines of the file';

sub plain_add ($a, $b, $out) {
    my ($x, $y) = thread_index();
    $out->[$x][$y] = $a->[$x][$y] + $b->[$x][$y];
}

subtest 'source => a code reference' => sub {
    for my $engine (both_engines) {
        my $r = PerCUDA->new(engine => $engine, source => \&plain_add, argument => [ out => 'f', a => 'f', b => 'f' ]);
        is $r->engine, $engine, "$engine: engine";
        is scalar $r->run($p, $q, []), [ [ 11, 22 ], [ 33, 44 ] ],
            "$engine: a named sub; the signature orders the arguments";

        my $scale = PerCUDA->new(
            engine   => $engine,
            argument => [ in => 'f', out => 'f', s => 'f' ],
            source   => sub ($in, $out, $s) { my $x = thread_index(); $out->[$x] = $in->[$x] * $s },
        );
        is scalar $scale->run([ 1, 2, 3 ], [], 0.5), [ 0.5, 1, 1.5 ], "$engine: an anonymous sub on one line";

        my $clamp = PerCUDA->new(
            engine   => $engine,
            argument => [ in => 'f', out => 'f' ],
            source   => sub ($in, $out) {
                my $x = thread_index();
                my $v = $in->[$x];
                $v = 0.0 if $v < 0;
                $out->[$x] = $v;
            },
        );
        is scalar $clamp->run([ -1, 2 ], []), [ 0, 2 ], "$engine: an anonymous sub over several lines";

        my $again = PerCUDA->new(engine => $engine, source => \&add, argument => [ a => 'f', b => 'f', out => 'f' ]);
        is scalar $again->run($p, $q, []), [ [ 11, 22 ], [ 33, 44 ] ], "$engine: a :GPU sub";
    }
    like dies { PerCUDA->new(source => \&plain_add, argument => [ a => 'f', b => 'f', out => 'f', n => 'i' ]) },
        qr/argument 'n' is not a parameter/, 'an argument that is not a parameter';
    like dies { PerCUDA->new(source => \&Scalar::Util::blessed, argument => []) }, qr/no body written in Perl/,
        'an XS sub';
    like dies { PerCUDA->new(source => [], argument => []) }, qr/must be a string or a code reference/,
        'another reference';
};

sub third : GPU({ precision => 'double' }) ($a) {
    my $x = thread_index();
    $a->[$x] = 1.0 / 3.0;
}

subtest 'review fixes' => sub {
    my ($one, $two) =
        (sub ($a, $b) { my $x = thread_index(); $b->[$x] = 1 }, sub ($a, $b) { my $x = thread_index(); $b->[$x] = 2 });
    like dies { PerCUDA->new(source => $two, engine => 'perl') }, qr/has 2 anonymous subs; put each kernel sub/,
        'two anonymous subs on one line are refused, not confused';

    my $third = PerCUDA->context(engine => 'perl', precision => 'single')->kernel(source => \&third);
    is { $third->arguments }, { a => 'double' }, 'the attribute of a :GPU sub wins over the context';
    is scalar $third->run([0]), [ 1 / 3 ], 'and computes in double precision';

    my $inf = PerCUDA->new(
        engine => 'perl',
        source => sub ($inf, $out) { my $x = thread_index(); $out->[$x] = $inf + 1 },
        size   => [2]
    );
    is scalar $inf->run(inf => 2, out => []), [ 3, 3 ], 'a named call whose first name looks like a number';
    is scalar $inf->run(2, []), [ 3, 3 ], 'and positional still works';

    my $merged = PerCUDA->new(source => \&with_constants, engine => 'perl', constant => { B => 3 });
    is scalar $merged->run([0]), [5], 'constants of new add to those of the attribute';
    my $named_x = PerCUDA->new(
        engine   => 'perl',
        argument => [ a => 'f' ],
        constant => { x => 5 },
        source   => 'my $i = thread_index(); $a->[$i] = $x + x;'
    );
    is scalar $named_x->run([ 0, 0 ]), [ 10, 10 ], 'a constant may be called x: it is not a reserved name';
    like dies {
        PerCUDA->new(
            argument => [ a => 'f' ],
            constant => { x => 5 },
            source   => 'my $x = thread_index(); $a->[$x] = 1;'
        )
    }, qr/constant \$x cannot be declared/, 'but it cannot also be declared as the thread index';
    my $types = PerCUDA::Analyzer::infer_types(
        PerCUDA::Parser->new(source => 'for (my $k = 0; $k < $n * 0.5; $k++) { $a->[$x] += 1; }')->parse,
        ['n'], 'f');
    is $types, { n => 'f' }, 'a loop bound with a float literal is not made an int';
};

our ($A, $B);    # constants given in options are $NAMEs

sub with_constants : GPU(float, { constant => { A => 2 } }) ($a) {
    my $x = thread_index();
    $a->[$x] = $A + $B;
}

{

    package PerCUDATest::RunTime;
    use PerCUDA;

    sub twice : GPU ($out, $in) {
        my $x = thread_index();
        $out->[$x] = $in->[$x] * 2;
    }

    sub keep_float : GPU({ precision => 'single' }) ($out) {
        my $x = thread_index();
        $out->[$x] = 1.0 / 3.0;
    }
    sub setup (@options) { return PerCUDA->configure(@options) }
    sub make  (@args)    { return PerCUDA->array(@args) }
}

subtest 'configure' => sub {
    for my $engine (both_engines) {
        PerCUDATest::RunTime::setup(engine => $engine, precision => 'double');
        is(PerCUDA->runner(\&PerCUDATest::RunTime::twice)->engine, $engine, "$engine: the configured engine");
        my $in  = PerCUDATest::RunTime::make([ 1, 2, 3 ]);
        my $out = PerCUDATest::RunTime::make(dims => [3]);
        is $in->type, 'double', "$engine: PerCUDA->array follows the configured precision";
        PerCUDATest::RunTime::twice($out, $in);
        is $out->get, [ 2, 4, 6 ], "$engine: device arrays of the configuration in a :GPU sub";
        is { PerCUDA->runner(\&PerCUDATest::RunTime::keep_float)->arguments }, { out => 'float' },
            "$engine: the attribute wins over the configuration";
    }
    PerCUDATest::RunTime::setup(engine => 'perl', precision => 'single');
    is { PerCUDA->runner(\&PerCUDATest::RunTime::twice)->arguments }, { out => 'float', in => 'float' },
        'configuring again rebuilds the runner';
    is(PerCUDA->runner(\&PerCUDATest::RunTime::twice)->engine, 'perl', 'on the new engine');

    require PerCUDATest::Configured;
    is { PerCUDA->runner(\&PerCUDATest::Configured::third)->arguments }, { out => 'double' },
        "use PerCUDA { ... } configures the package";
    is(PerCUDATest::Configured::array_here(dims => [1])->engine, 'perl',   'PerCUDA->array in that package');
    is(PerCUDA->runner(\&add)->engine, (gpu_available() ? 'gpu' : 'perl'), 'other packages are not affected');

    # PerCUDA->array in main (not configured) does not configure main's :GPU subs
    PerCUDA->array(dims => [1]);
    is(PerCUDA->runner(\&add_perl)->engine, 'perl', 'PerCUDA->array leaves the :GPU subs of the package alone');

    like dies { PerCUDATest::RunTime::setup(argument => [ n => 'int' ]) }, qr/configure cannot set 'argument'/,
        'per-kernel options are refused';
    like dies { PerCUDATest::RunTime::setup('engine') },             qr/name => value pairs/, 'an odd list';
    like dies { PerCUDATest::RunTime::setup([ engine => 'perl' ]) }, qr/name => value pairs/, 'an array reference';

    my ($names, $body) = PerCUDA::Kernel::extract_sub("sub last (\$a) {\n    \$a->[\$#{\$a}] = 1;\n}\n", 'last', 2);
    like $body, qr/\$#\{\$a\}\] = 1;/, '$#{...} is not a comment';
    my $grid = PerCUDA->new(
        engine => 'perl',
        source => sub ($t, $in, $out) { my $x = thread_index(); $out->[$x] = $in->[$x] + $t->[0][0] }
    );
    is scalar $grid->run([ [] ], [ 1, 2 ], []), [ 1, 2 ], 'an array with an empty row is not the grid';
};

{

    package PerCUDATest::Functions;
    use PerCUDA qw(configure array runner);

    sub inc : GPU ($a) {
        my $x = thread_index();
        $a->[$x] += 1;
    }

    sub check_functions () {
        configure(engine => 'perl', precision => 'double');
        my $a = array([ 1, 2 ]);
        inc($a);
        return (runner(\&inc)->engine, $a->type, $a->get);
    }

    sub qualified () {
        PerCUDA::configure({ engine => 'perl' });
        return (PerCUDA::array(dims => [1])->type, PerCUDA::runner(\&inc)->engine);
    }
}

subtest 'functions' => sub {
    my ($engine, $type, $values) = PerCUDATest::Functions::check_functions();
    is [ $engine, $type, $values ], [ 'perl', 'double', [ 2, 3 ] ], 'imported configure, array and runner';
    is [ PerCUDATest::Functions::qualified() ], [ 'float', 'perl' ], 'PerCUDA::configure, ::array, ::runner';
    like dies { eval q{ package T4; use PerCUDA qw(nope); 1 } or die $@ }, qr/unknown import 'nope'/, 'unknown import';
    like dies { eval q{ package T5; use strict; use PerCUDA (); sub k :GPU ($a) { 1 } 1 } or die $@ },
        qr/Invalid CODE attribute: GPU/, 'use PerCUDA () does not enable :GPU';
    ok !T5->can('MODIFY_CODE_ATTRIBUTES'), 'and installs nothing';

    local $ENV{PERCUDA_ENGINE} = 'perl';
    is(PerCUDA->new(argument => [ a => 'f' ], source => 'my $x = thread_index(); $a->[$x] = 1;')->engine,
        'perl', 'PERCUDA_ENGINE');
    is(PerCUDA->context->engine, 'perl', 'PERCUDA_ENGINE in a context');
};

use constant SIZE => 3;
my $side_effects = 0;
sub NOT_A_CONSTANT () { $side_effects++; return 1 }

sub sized : GPU({ size => [SIZE] }) ($out) {    # a constant in the options
    my $x = thread_index();
    $out->[$x] = $x;
}

sub commented : GPU (
    $in,     # input
    $out,    # output
    )
{
    my $x = thread_index();
    $out->[$x] = $in->[$x];
}

subtest 'review fixes (2)' => sub {
    is scalar sized([]), [ 0, 1, 2 ], 'the options see the constants of the package';
    is scalar commented([ 1, 2 ], []), [ 1, 2 ], 'comments in a signature';
    like dies {
        PerCUDA->new(engine => 'perl', source => sub ($a) { my $x = thread_index(); $a->[$x] = NOT_A_CONSTANT })
    }, qr/unknown name 'NOT_A_CONSTANT'/, 'a sub with an empty prototype is not a constant';
    is $side_effects, 0, 'and is not called';
    like dies { PerCUDA::Kernel::parse_attribute('f, { size => [ $m ] }') }, qr/cannot read the :GPU options/,
        'a variable in the options is an error';
    like dies { PerCUDA->context(engine => 'perl', precision => 'doubel') }, qr/precision must be/,
        'a context checks precision';
    like dies { PerCUDA->context(engine => 'perl', precison => 'double') }, qr/unknown option 'precison'/,
        'and its option names';

    our $K;    # a context constant, written $K in a sub
    my $pc     = PerCUDA->context(engine => 'perl', argument => [ n => 'int' ], constant => { K => 7 });
    my $uses_k = $pc->kernel(
        source   => sub ($out, $n) { my $x = thread_index(); $out->[$x] = $K + $n },
        argument => [ out => 'double' ]
    );
    is [ $uses_k->arguments ], [ out => 'double', n => 'int' ], 'context argument types merge with those of the kernel';
    is scalar $uses_k->run([ 0, 0 ], 1), [ 8, 8 ], 'a context constant';
    my $own_k = $pc->kernel(source => sub ($out, $K) { my $x = thread_index(); $out->[$x] = $K });
    is scalar $own_k->run([0], 3), [3], 'a parameter $K does not clash with the context constant K';
    my $outer = PerCUDA->new(
        engine => 'perl',
        source => sub ($v, $c) { my ($x, $y) = thread_index(); $c->[$x][$y] = $v->[$x] }
    );
    like dies { $outer->run([ 1, 2, 3 ], []) }, qr/no array with 2 or more dimensions was passed with elements/,
        'a 1-D array is not taken as the grid of a 2-D kernel';
};

use constant N => 4;

sub both_n ($out, $N) {    # N (a constant) and $N (a parameter) are different names, as in Perl
    my $x = thread_index();
    $out->[$x] = N + $N;
}

subtest 'constants and variables' => sub {
    is scalar PerCUDA->new(source => \&both_n, engine => 'perl', size => [2])->run([], 10), [ 14, 14 ],
        'use constant N does not replace $N';
    like dies {
        PerCUDA->new(
            argument => [ a => 'f' ],
            constant => { K => '1e400' },
            source   => 'my $x = thread_index(); $a->[$x] = K;'
        )
    }, qr/constant 'K' must be a finite number/, 'an infinite constant';
    like dies { PerCUDA->new(argument => [ a => 'f' ], source => 'my $x = thread_index(); $a->[$x] = 1 +;') },
        qr/at \Q${\__FILE__}\E line/,
        'kernel errors point at the caller of new';
};

subtest 'source file read after chdir' => sub {
    require File::Temp;
    my $dir = File::Temp->newdir;
    my $cwd = Cwd::getcwd();
    chdir $dir or die $!;
    my $ok = eval { PerCUDA->new(source => \&both_n, engine => 'perl', size => [1]); 1 };
    chdir $cwd or die $!;
    ok $ok, 'a relative source path still resolves' or diag $@;
};

subtest 'kernel subs in a module' => sub {
    require PerCUDATest::KernelSubs;
    my $out = [ 0, 0, 0 ];
    is [ PerCUDATest::KernelSubs::scale([ 1, 2, 3 ], $out, 2) ], [], 'retval => [] returns nothing';
    is $out, [ 0, 0, 0 ], 'and leaves the caller\'s array alone';
    is scalar PerCUDATest::KernelSubs::twice([], [ 1, 2, 3 ]), [ 2, 4, 6 ],
        'the output first: the grid comes from the input';
};

subtest 'attribute' => sub {
    my ($types, $options) = PerCUDA::Kernel::parse_attribute(q{f, i, d, { precision => 'double', engine => perl }});
    is $types,                                                                 [qw(f i d)],   'types (short codes)';
    is [ (PerCUDA::Kernel::parse_attribute('float, int, double, f'))[0]->@* ], [qw(f i d f)], 'type names';
    is $options, { precision => 'double', engine => 'perl' },                  'options';
    is [ PerCUDA::Kernel::parse_attribute('') ],                [ [], {} ],             'nothing: default types';
    is [ PerCUDA::Kernel::parse_attribute(q{{ check => 1 }}) ], [ [], { check => 1 } ], 'options only';
    like dies { PerCUDA::Kernel::parse_attribute('f, x') },              qr/expects types/,       'bad type';
    like dies { PerCUDA::Kernel::parse_attribute('f, { color => 1 }') }, qr/unknown :GPU option/, 'bad option';
    like dies { PerCUDA::Kernel::parse_attribute('f, check => 1') }, qr/one hash after the types/,
        'options not in a hash';
    like dies { PerCUDA::Kernel::parse_attribute('f, { check => 1 }, { perl_jobs => 1 }') },
        qr/one hash after the types/,
        'two hashes';

    ($types, $options) = PerCUDA::Kernel::parse_attribute(<<'ATTR');
float, int, {
    precision => double, constant => { PI => 3.14159, N => -2 },
    size => [ 'n', 3 ], retval => [qw(out)], check => 1
}
ATTR
    is $types, [qw(f i)], 'types over several lines';
    is $options,
        {
        precision => 'double',
        constant  => { PI => 3.14159, N => -2 },
        size      => [ 'n', 3 ],
        retval    => ['out'],
        check     => 1
        },
        'options written as in Perl';
    like dies { PerCUDA::Kernel::parse_attribute('f, { size => [ }') }, qr/cannot read the :GPU options/, 'not Perl';
};

subtest 'default types' => sub {
    my $types = sub ($source, @names) {
        my $ast = PerCUDA::Parser->new(source => $source)->parse;
        return PerCUDA::Analyzer::infer_types($ast, \@names, 'f', { n => 1 });
    };
    is $types->('$out->[$x] = $in->[$x] * $s;', qw(out in s)), { out => 'f', in => 'f', s => 'f' },
        'arrays and a factor';
    is $types->('$out->[$x] = $in->[$x * $stride + $off];', qw(stride off)), { stride => 'i', off => 'i' },
        'used in an index';
    is $types->('$out->[$x] = $in->[int($p * $x)];',    qw(p)),      { p => 'f' },              'not through int()';
    is $types->('$out->[$x] = $x % $k + ($x & $mask);', qw(k mask)), { k => 'i', mask => 'i' }, '% and bit operations';
    is $types->('$out->[$x] = random($seed, $x);',                     qw(seed)), { seed => 'i' }, 'random()';
    is $types->('for my $j (0 .. $m - 1) { $out->[$x] += 1; }',        qw(m)),    { m    => 'i' }, 'a foreach bound';
    is $types->('for (my $j = 0; $j < $m; $j++) { $out->[$x] += 1; }', qw(m)), { m => 'i' }, 'a C-style int loop bound';
    is $types->('my $row = $x * $w; $out->[$row + 1] = 1;', qw(w)), { w => 'i' }, 'through a local used as an index';
    is $types->('return if $x >= $m; $out->[$x] = 1;',      qw(m)), { m => 'f' },
        'a threshold compared with a thread index stays a float';
    is $types->('my $v = $s * 2; $out->[$x] = $v;', qw(s)), { s => 'f' }, 'a local that is not an int passes nothing';
    is $types->('for (my $t = 0.0; $t < $m; $t += 0.1) { $out->[$x] += 1; }', qw(m)), { m => 'f' }, 'a float loop';
    is $types->('$out->[$x] = $x;',                                           qw(n)), { n => 'i' }, 'a size';
    is $types->('$a->[$x] = $a->[$x] % 2;', qw(a)), { a => 'f' }, 'an array is never int by default';

    sub typed : GPU ($out, $in, $k, $seed, $s) {
        my $x = thread_index();
        $out->[$x] = $in->[ $x % $k ] * $s + random($seed, $x);
    }
    is [ PerCUDA->runner(\&typed)->arguments ],
        [ out => 'float', in => 'float', k => 'int', seed => 'int', s => 'float' ], '$runner->arguments shows them';
    my $partial = PerCUDA->new(source => \&typed, argument => [ out => 'double' ]);
    is [ $partial->arguments ], [ out => 'double', in => 'float', k => 'int', seed => 'int', s => 'float' ],
        'argument gives some types, the rest are defaults';

    sub typed_double : GPU({ precision => 'double', argument => [ k => 'float' ] }) ($out, $k) {
        my $x = thread_index();
        $out->[$x] = $k;
    }
    is [ PerCUDA->runner(\&typed_double)->arguments ], [ out => 'double', k => 'float' ],
        'precision double, and argument in the :GPU options';
    like dies { PerCUDA->new(argument => [ a => undef ], source => 'my $x = thread_index(); $a->[$x] = 1;') },
        qr/unknown type/,
        'a string kernel needs every type';

    my $inherited = PerCUDA->new(source => \&typed_double);
    is [ $inherited->arguments ], [ out => 'double', k => 'float' ], 'new(source => \&gpu_sub) takes its attribute';
    my $overridden = PerCUDA->new(source => \&typed_double, precision => 'single', argument => [ k => 'int' ]);
    is [ $overridden->arguments ], [ out => 'float', k => 'int' ], 'and the arguments of new win';
};

use constant { K => 10, HALF => 0.5 };
our $OFFSET;    # a constant given in the attribute is a $NAME, so it needs declaring

sub with_options : GPU({ constant => { OFFSET => 100 }, size => [ 'n', 2 ] }) ($out, $n) {
    my ($x, $y) = thread_index();
    $out->[$x][$y] = $x * K + $y + $OFFSET;
}
is scalar with_options([], 2), [ [ 100, 101 ], [ 110, 111 ] ], 'use constant, constant and a list size in :GPU';

sub halves ($in, $out) { my $x = thread_index(); $out->[$x] = $in->[$x] * HALF }
is scalar PerCUDA->new(source => \&halves, argument => [ in => 'float', out => 'float' ])->run([ 2, 4 ], []), [ 1, 2 ],
    'use constant with source => \&sub';
like dies { PerCUDA->new(argument => [ out => 'float' ], source => 'my $x = thread_index(); $out->[$x] = NOPE;') },
    qr/unknown name 'NOPE'/, 'an unknown bareword';
is scalar PerCUDA->new(
    argument => [ out => 'float' ],
    constant => { C => 2 },
    size     => [2],
    source   => 'my $x = thread_index(); $out->[$x] = C;'
)->run([]), [ 2, 2 ], 'constant => { C => ... } as a bareword in a string kernel';

subtest 'source of the sub' => sub {
    my $src = <<'PERL';
my $unrelated = 1;
sub k :GPU(f) :lvalue ($in) {
    # } in a comment
    if ($in->[$x]) { $in->[$x] = 1; }
}
PERL
    my ($names, $body) = PerCUDA::Kernel::extract_sub($src, 'k', 2);
    is $names, ['in'], 'parameters';
    like $body, qr/\A\n\n    # \} in a comment\n\s*if .*\}\n\z/s, 'the body, without its braces, at its line';
    my $multi = "sub m\n    :GPU(f, f)\n    (\$in,\n     \$out)\n{\n    \$out->[\$x] = \$in->[\$x];\n}\n";
    ($names, $body) = PerCUDA::Kernel::extract_sub($multi, 'm', 6);
    is $names, [qw(in out)],                                  'a declaration over several lines';
    is $body,  "\n\n\n\n\n    \$out->[\$x] = \$in->[\$x];\n", 'its body';
    like dies { PerCUDA::Kernel::extract_sub("sub k :GPU(f) { 1 }\n", 'k', 1) }, qr/needs a signature/, 'no signature';
    like dies { PerCUDA::Kernel::extract_sub("sub k :GPU(f) (\$a = 1) { 1 }\n", 'k', 1) }, qr/plain scalar/,
        'a default value';
    like dies { PerCUDA::Kernel::extract_sub("sub k :GPU(f) (\$a) { 1 \n", 'k', 1) }, qr/unbalanced/, 'unbalanced';
};

like dies {
    eval q{ package T1; use PerCUDA; sub k :GPU(f) ($a) { my $x = thread_index(); $a->[$x] = 1; } 1 }
        or die $@
}, qr/cannot read the source of sub k/, 'a sub compiled from a string has no source file';
like dies { eval q{ package T2; use PerCUDA ':gpu'; 1 } or die $@ }, qr/unknown import ':gpu'/, 'unknown import';

is \@warnings, [], 'no warnings';

done_testing;
