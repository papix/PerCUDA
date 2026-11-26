package PerCUDA::CodeGen::CUDA;
use v5.40;
use PerCUDA::Analyzer qw(is_pure error);
use PerCUDA::Math;

# Emits the kernel as CUDA C++ for NVRTC. Arrays are flattened in row-major
# order. The last Perl axis is mapped to CUDA's x axis,
# so that neighbouring threads touch neighbouring elements (coalesced access).

our $KERNEL_NAME = 'percuda_kernel';

my %CTYPE = (i => 'int', f => 'float', d => 'double');

my $PRELUDE = <<'C';
static __device__ inline long long percuda_mod(long long a, long long b) {
    if (b == 0) return 0;
    long long r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
static __device__ inline int percuda_mod_i(int a, int b) {
    if (b == 0 || b == -1) return 0;    /* b == -1 also avoids INT_MIN % -1 */
    int r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
static __device__ inline int percuda_powi_i(int x, int n) {
    int r = 1;
    while (n > 0) { if (n & 1) r *= x; x *= x; n >>= 1; }
    return r;
}
static __device__ inline float percuda_powi_f(float x, int n) {
    float r = 1.0f;
    unsigned m = n < 0 ? -(unsigned)n : (unsigned)n;
    while (m) { if (m & 1) r *= x; x *= x; m >>= 1; }
    return n < 0 ? 1.0f / r : r;
}
static __device__ inline double percuda_powi_d(double x, int n) {
    double r = 1.0;
    unsigned m = n < 0 ? -(unsigned)n : (unsigned)n;
    while (m) { if (m & 1) r *= x; x *= x; m >>= 1; }
    return n < 0 ? 1.0 / r : r;
}
static __device__ inline int percuda_abs_i(int x) { return x < 0 ? -x : x; }
/* random(): a counter-based generator; PerCUDA::Runtime::random_* is the same */
static __device__ inline unsigned int percuda_mix(unsigned int x) {    /* triple32 (Chris Wellons) */
    x ^= x >> 17; x *= 0xed5ad4bbu; x ^= x >> 11; x *= 0xac4c1b51u; x ^= x >> 15; x *= 0x31848babu; x ^= x >> 14;
    return x;
}
static __device__ inline unsigned int percuda_hash(int n, int k0, int k1, int k2, int k3) {
    unsigned int h = percuda_mix(0x9e3779b9u + (unsigned int)n);
    h = percuda_mix(h ^ (unsigned int)k0);
    if (n > 1) h = percuda_mix(h ^ (unsigned int)k1);
    if (n > 2) h = percuda_mix(h ^ (unsigned int)k2);
    if (n > 3) h = percuda_mix(h ^ (unsigned int)k3);
    return h;
}
static __device__ inline float percuda_random_f(int n, int k0, int k1, int k2, int k3) {
    return (float)(percuda_hash(n, k0, k1, k2, k3) >> 8) * 5.9604644775390625e-08f;    /* 2^-24 */
}
static __device__ inline double percuda_random_d(int n, int k0, int k1, int k2, int k3) {
    unsigned int h = percuda_hash(n, k0, k1, k2, k3);
    unsigned int l = percuda_mix(h ^ 0x85ebca6bu);
    return ((double)(h >> 5) * 67108864.0 + (double)(l >> 6)) * 1.1102230246251565e-16;    /* 2^-53 */
}
C

# mode => 'exact' uses PerCUDA::Math instead of CUDA's math functions, so that
# the perl engine can reproduce the results bit for bit.
#
# check => 1 routes every array access through percuda_at, which records
# the first out-of-range index in percuda_err (read back by the GPU engine)
# and points the access at a scratch variable instead.
sub new ($class, $analyzer, %opt) {
    return bless { an => $analyzer, portable => ($opt{mode} // 'compat') eq 'exact', check => $opt{check} ? 1 : 0 },
        $class;
}

my $CHECK_PRELUDE = <<'C';
static __device__ void percuda_fail(int *err, int id, int line, int axis, int index, int x, int y, int z) {
    if (atomicCAS(err, 0, 1) == 0) {
        err[1] = id; err[2] = line; err[3] = axis; err[4] = index; err[5] = x; err[6] = y; err[7] = z;
    }
}
template <typename T>
static __device__ T *percuda_at(T *base, T *scratch, int *err, int id, int line, int x, int y, int z, int rank,
                                int i0, int i1, int i2, int d0, int d1, int d2) {
    int idx[3] = { i0, i1, i2 };
    int dim[3] = { d0, d1, d2 };
    for (int k = 0; k < rank; k++) {
        if (idx[k] < 0 || idx[k] >= dim[k]) {
            percuda_fail(err, id, line, k, idx[k], x, y, z);
            *scratch = 0;
            return scratch;
        }
    }
    int flat = i0;
    if (rank > 1) flat = flat * d1 + i1;
    if (rank > 2) flat = flat * d2 + i2;
    return base + flat;
}
C

sub generate ($self) {
    my $an   = $self->{an};
    my @axes = $an->axes;

    my @params;
    for my $p (@{ $an->kernel_params($self->{check}) }) {
        if ($p->{kind} eq 'error') {
            push @params, 'int *percuda_err';
            next;
        }
        my $ctype = $CTYPE{ $p->{type} };
        if ($p->{kind} eq 'array') {
            my $arg   = $an->args->[ $p->{arg} ];
            my $const = $arg->{written} ? '' : 'const ';
            push @params, "$const$ctype *__restrict__ v_$p->{name}";
        }
        elsif ($p->{kind} eq 'size') {
            push @params, "int percuda_$p->{name}";
        }
        else {
            push @params, "$ctype " . ($p->{kind} eq 'scalar' ? "v_$p->{name}" : $p->{name});
        }
    }

    my @prologue;
    for my $i (0 .. $#axes) {
        my $cuda = (qw(x y z))[ $#axes - $i ];
        push @prologue, "int percuda_$axes[$i] = blockIdx.$cuda * blockDim.$cuda + threadIdx.$cuda;";
    }
    push @prologue, 'if (' . join(' || ', map { "percuda_$_ >= percuda_size_$_" } @axes) . ') return;';
    push @prologue, 'int percuda_scratch_i; float percuda_scratch_f; double percuda_scratch_d;' if $self->{check};

    my $body     = join '', map { $self->_stmt($_, 1) } @{ $an->ast->{body} };
    my $prologue = join '', map { "    $_\n" } @prologue;
    my $prelude  = $PRELUDE;
    $prelude .= PerCUDA::Math::c_prelude() if $self->{portable};
    $prelude .= $CHECK_PRELUDE             if $self->{check};
    return "$prelude\nextern \"C\" __global__ void $KERNEL_NAME(" . join(', ', @params) . ") {\n$prologue$body}\n";
}

sub _ind ($level) { return '    ' x $level }

sub _block ($self, $block, $level) {
    my $inner = join '', map { $self->_stmt($_, $level + 1) } @{ $block->{body} };
    return "{\n$inner" . _ind($level) . "}";
}

sub _stmt ($self, $s, $level) {
    my $ind = _ind($level);
    my $t   = $s->{t};
    return '' if $t eq 'index';    # the thread index and grid size are set before the body
    if ($t eq 'my') {
        my $type = $s->{sym}{type};
        my $init = $s->{init} ? $self->_cast($type, $s->{init}) : '0';
        return "$ind$CTYPE{$type} $s->{sym}{cname} = $init;\n";
    }
    return $ind . $self->_expr($s->{e}) . ";\n"    if $t eq 'expr';
    return $ind . $self->_block($s, $level) . "\n" if $t eq 'block';
    if ($t eq 'if') {
        my @parts = map { 'if (' . $self->_expr($_->[0]) . ') ' . $self->_block($_->[1], $level) } @{ $s->{conds} };
        my $out   = $ind . join(' else ', @parts);
        $out .= ' else ' . $self->_block($s->{else}, $level) if $s->{else};
        return "$out\n";
    }
    if ($t eq 'while') {
        return $ind . 'while (' . $self->_expr($s->{cond}) . ') ' . $self->_block($s->{body}, $level) . "\n";
    }
    if ($t eq 'for') {
        my $init = $s->{init} ? $self->_stmt($s->{init}, $level + 1) : '';
        my $cond = $s->{cond} ? $self->_expr($s->{cond})             : '1';
        my $step = $s->{step} ? $self->_expr($s->{step})             : '';
        return
              "$ind\{\n$init"
            . _ind($level + 1)
            . "for (; $cond; $step) "
            . $self->_block($s->{body}, $level + 1)
            . "\n$ind}\n";
    }
    if ($t eq 'foreach') {

        # Perl evaluates the range once and the loop variable is a copy per iteration
        my $id = $self->{an}->new_id;
        my ($lo, $hi, $it) = ("percuda_lo$id", "percuda_hi$id", "percuda_it$id");
        my ($i1, $i2) = (_ind($level + 1), _ind($level + 2));
        return
              "$ind\{\n"
            . "${i1}int $lo = "
            . $self->_cast('i', $s->{from}) . ";\n"
            . "${i1}int $hi = "
            . $self->_cast('i', $s->{to}) . ";\n"
            . "${i1}for (int $it = $lo; $it <= $hi; $it++) {\n"
            . "${i2}$CTYPE{$s->{sym}{type}} $s->{sym}{cname} = $it;\n" . "$i2"
            . $self->_block($s->{body}, $level + 2) . "\n"
            . "$i1}\n$ind}\n";
    }
    return "${ind}break;\n"    if $t eq 'last';
    return "${ind}continue;\n" if $t eq 'next';
    return "${ind}return;\n"   if $t eq 'return';
    error("unsupported statement '$t'", $s);
}

sub _cast ($self, $type, $e) {
    my $c = $self->_expr($e);
    return $self->{an}->type_of($e) eq $type ? $c : "(($CTYPE{$type})($c))";
}

sub _num ($self, $e) {
    return sprintf('%d', $e->{value}) unless $e->{float};
    my $double = $self->{an}->real eq 'd';
    my $s      = sprintf($double ? '%.17g' : '%.9g', $e->{value});
    $s .= '.0' unless $s =~ /[.e]/;
    return $double ? $s : "${s}f";
}

sub _expr ($self, $e) {
    my $t = $e->{t};
    return $self->_num($e)  if $t eq 'num';
    return $e->{sym}{cname} if $t eq 'var';
    return $self->_elem($e) if $t eq 'elem';
    if ($t eq 'un') {
        return '(-' . $self->_expr($e->{e}) . ')' if $e->{op} eq '-';
        return '(!' . $self->_expr($e->{e}) . ')' if $e->{op} eq '!';
        return '(~' . $self->_cast('i', $e->{e}) . ')';
    }
    return "($e->{op}" . $self->_expr($e->{e}) . ')' if $t eq 'pre';
    return '(' . $self->_expr($e->{e}) . "$e->{op})" if $t eq 'post';
    if ($t eq 'cond') {
        my $type = $self->{an}->type_of($e);
        return
              '('
            . $self->_expr($e->{c}) . ' ? '
            . $self->_cast($type, $e->{a}) . ' : '
            . $self->_cast($type, $e->{b}) . ')';
    }
    return $self->_assign($e) if $t eq 'assign';
    return $self->_call($e)   if $t eq 'call';
    return $self->_bin($e)    if $t eq 'bin';
    error("unsupported expression '$t'");
}

sub _elem ($self, $e) {
    my $name = $e->{name};
    my @idx  = map { $self->_cast('i', $_) } @{ $e->{idx} };
    if ($self->{check}) {
        my $an    = $self->{an};
        my $rank  = @idx;
        my $type  = $e->{sym}{type};
        my ($id)  = grep { $an->args->[$_]{name} eq $name } 0 .. $#{ $an->args };
        my @t     = map  { $_ < $an->dim ? "percuda_$PerCUDA::Analyzer::AXES[$_]" : '0' } 0 .. 2;
        my @i     = map  { $idx[$_] // '0' } 0 .. 2;
        my @d     = map  { $_ < $rank ? "d_${name}_$_" : '0' } 0 .. 2;
        my $ctype = $CTYPE{$type};
        return
              "(*percuda_at(($ctype *)v_$name, &percuda_scratch_$type, percuda_err, $id, "
            . ($e->{line} // 0)
            . ", @{[ join ', ', @t ]}, $rank, @{[ join ', ', @i ]}, @{[ join ', ', @d ]}))";
    }
    my $flat = shift @idx;
    my $d    = 1;
    for my $i (@idx) {
        $flat = "($flat) * d_${name}_$d + ($i)";
        $d++;
    }
    return "v_${name}[$flat]";
}

sub _assign ($self, $e) {
    my $lhs  = $self->_expr($e->{lhs});
    my $type = $self->{an}->type_of($e->{lhs});
    my $op   = $e->{op};
    return "($lhs = " . $self->_cast($type, $e->{rhs}) . ')'                 if $op eq '=';
    return "($lhs ? $lhs : ($lhs = " . $self->_cast($type, $e->{rhs}) . '))' if $op eq '||=';
    return "($lhs ? ($lhs = " . $self->_cast($type, $e->{rhs}) . ") : $lhs)" if $op eq '&&=';
    my $bin = { t => 'bin', op => ($op =~ s/=\z//r), l => $e->{lhs}, r => $e->{rhs} };
    return "($lhs = " . $self->_cast($type, $bin) . ')';
}

sub _call ($self, $e) {
    my $arg  = $e->{args}[0];
    my $type = $self->{an}->type_of($e);
    if ($e->{name} eq 'sqrt') {
        return ($type eq 'd' ? 'sqrt(' : 'sqrtf(') . $self->_cast($type, $arg) . ')';
    }
    if ($e->{name} eq 'abs') {
        my $fn = { i => 'percuda_abs_i', f => 'fabsf', d => 'fabs' }->{$type};
        return "$fn(" . $self->_expr($arg) . ')';
    }
    return $self->_cast('i', $arg) if $e->{name} eq 'int';
    if ($e->{name} eq 'random') {
        my @keys = map { $self->_cast('i', $_) } @{ $e->{args} };
        push @keys, '0' while @keys < 4;
        return "percuda_random_$type(" . scalar(@{ $e->{args} }) . ', ' . join(', ', @keys) . ')';
    }

    # floor and ceil are exact; exp, log, sin, cos and atan2 come from CUDA
    # (fastest) or from PerCUDA::Math (reproducible by the perl engine)
    my $name = $e->{name};
    my $args = join ', ', map { $self->_cast($type, $_) } @{ $e->{args} };
    return "percuda_${name}_$type($args)" if $self->{portable} && grep { $_ eq $name } @PerCUDA::Math::FUNCTIONS;
    return ($type eq 'f' ? "${name}f" : $name) . "($args)";
}

sub _bin ($self, $e) {
    my $op   = $e->{op};
    my $type = $self->{an}->type_of($e);
    my ($l, $r) = ($e->{l}, $e->{r});

    return '(' . $self->_cast($type, $l) . ' / ' . $self->_cast($type, $r) . ')' if $op eq '/';
    if ($op eq '%') {

        # 64-bit division is emulated on GPUs; int operands need only 32 bits
        if ($self->{an}->type_of($l) eq 'i' && $self->{an}->type_of($r) eq 'i') {
            return 'percuda_mod_i(' . $self->_expr($l) . ', ' . $self->_expr($r) . ')';
        }
        return '((int)percuda_mod((long long)(' . $self->_expr($l) . '), (long long)(' . $self->_expr($r) . ')))';
    }
    if ($op eq '**') {
        if ($self->{an}->type_of($r) eq 'i') {    # repeated multiplication, identical on both engines
            return "percuda_powi_$type(" . $self->_cast($type, $l) . ', ' . $self->_expr($r) . ')';
        }
        my $fn = $self->{portable} ? "percuda_pow_$type" : $type eq 'f' ? 'powf' : 'pow';
        return "$fn(" . $self->_cast($type, $l) . ', ' . $self->_cast($type, $r) . ')';
    }
    if ($op eq '&' || $op eq '|' || $op eq '^' || $op eq '<<' || $op eq '>>') {
        return '(' . $self->_cast('i', $l) . " $op " . $self->_cast('i', $r) . ')';
    }
    if (($op eq '||' || $op eq '&&') && is_pure($e)) {

        # Perl returns the deciding operand, not a boolean
        my ($lc, $lt, $rt) = ($self->_expr($l), $self->_cast($type, $l), $self->_cast($type, $r));
        return $op eq '||' ? "($lc ? $lt : $rt)" : "($lc ? $rt : $lt)";
    }
    return '(' . $self->_expr($l) . " $op " . $self->_expr($r) . ')';
}

1;
