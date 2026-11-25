package PerCUDA::CodeGen::CUDA;
use v5.40;
use PerCUDA::Analyzer qw(is_pure error);

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
C

sub new ($class, $analyzer) { bless { an => $analyzer }, $class }

sub generate ($self) {
    my $an   = $self->{an};
    my @axes = $an->axes;

    my @params;
    for my $p (@{ $an->kernel_params }) {
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

    my $body     = join '', map { $self->_stmt($_, 1) } @{ $an->ast->{body} };
    my $prologue = join '', map { "    $_\n" } @prologue;
    return "$PRELUDE\nextern \"C\" __global__ void $KERNEL_NAME(" . join(', ', @params) . ") {\n$prologue$body}\n";
}

sub _ind ($level) { '    ' x $level }

sub _block ($self, $block, $level) {
    my $inner = join '', map { $self->_stmt($_, $level + 1) } @{ $block->{body} };
    return "{\n$inner" . _ind($level) . "}";
}

sub _stmt ($self, $s, $level) {
    my $ind = _ind($level);
    my $t   = $s->{t};
    if ($t eq 'my') {
        my $type = $s->{sym}{type};
        my $init = $s->{init} ? $self->_cast($type, $s->{init}) : '0';
        return "$ind$CTYPE{$type} $s->{sym}{cname} = $init;\n";
    }
    return $ind . $self->_expr($s->{e}) . ";\n" if $t eq 'expr';
    return $ind . $self->_block($s, $level) . "\n" if $t eq 'block';
    if ($t eq 'if') {
        my @parts = map { 'if (' . $self->_expr($_->[0]) . ') ' . $self->_block($_->[1], $level) } @{ $s->{conds} };
        my $out = $ind . join(' else ', @parts);
        $out .= ' else ' . $self->_block($s->{else}, $level) if $s->{else};
        return "$out\n";
    }
    if ($t eq 'while') {
        return $ind . 'while (' . $self->_expr($s->{cond}) . ') ' . $self->_block($s->{body}, $level) . "\n";
    }
    if ($t eq 'for') {
        my $init = $s->{init} ? $self->_stmt($s->{init}, $level + 1) : '';
        my $cond = $s->{cond} ? $self->_expr($s->{cond}) : '1';
        my $step = $s->{step} ? $self->_expr($s->{step}) : '';
        return "$ind\{\n$init" . _ind($level + 1) . "for (; $cond; $step) "
            . $self->_block($s->{body}, $level + 1) . "\n$ind}\n";
    }
    if ($t eq 'foreach') {
        # Perl evaluates the range once and the loop variable is a copy per iteration
        my $id = $self->{an}->new_id;
        my ($lo, $hi, $it) = ("percuda_lo$id", "percuda_hi$id", "percuda_it$id");
        my ($i1, $i2) = (_ind($level + 1), _ind($level + 2));
        return "$ind\{\n"
            . "${i1}int $lo = " . $self->_cast('i', $s->{from}) . ";\n"
            . "${i1}int $hi = " . $self->_cast('i', $s->{to}) . ";\n"
            . "${i1}for (int $it = $lo; $it <= $hi; $it++) {\n"
            . "${i2}$CTYPE{$s->{sym}{type}} $s->{sym}{cname} = $it;\n"
            . "$i2" . $self->_block($s->{body}, $level + 2) . "\n"
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
    my $s = sprintf($double ? '%.17g' : '%.9g', $e->{value});
    $s .= '.0' unless $s =~ /[.e]/;
    return $double ? $s : "${s}f";
}

sub _expr ($self, $e) {
    my $t = $e->{t};
    return $self->_num($e) if $t eq 'num';
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
        return '(' . $self->_expr($e->{c}) . ' ? ' . $self->_cast($type, $e->{a}) . ' : ' . $self->_cast($type, $e->{b}) . ')';
    }
    return $self->_assign($e) if $t eq 'assign';
    return $self->_call($e)   if $t eq 'call';
    return $self->_bin($e)    if $t eq 'bin';
    error("unsupported expression '$t'");
}

sub _elem ($self, $e) {
    my $name = $e->{name};
    my @idx  = map { $self->_cast('i', $_) } @{ $e->{idx} };
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
    return "($lhs = " . $self->_cast($type, $e->{rhs}) . ')' if $op eq '=';
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
    error("unsupported function '$e->{name}'");
}

sub _bin ($self, $e) {
    my $op   = $e->{op};
    my $type = $self->{an}->type_of($e);
    my ($l, $r) = ($e->{l}, $e->{r});

    return '(' . $self->_cast($type, $l) . ' / ' . $self->_cast($type, $r) . ')' if $op eq '/';
    if ($op eq '%') {
        return '((int)percuda_mod((long long)(' . $self->_expr($l) . '), (long long)(' . $self->_expr($r) . ')))';
    }
    if ($op eq '**') {
        error("'**' needs an integer exponent") if $self->{an}->type_of($r) ne 'i';
        return "percuda_powi_$type(" . $self->_cast($type, $l) . ', ' . $self->_expr($r) . ')';
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
