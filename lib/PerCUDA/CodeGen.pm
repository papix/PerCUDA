package PerCUDA::CodeGen;
use v5.40;
use Carp ();
use List::Util qw(max);

# Translates a kernel AST (PerCUDA::Parser) into a C function for the NVPTX
# target of clang. Every kernel parameter is a device pointer:
#
#   void percuda_kernel(T0 *p0, T1 *p1, ..., int *percuda_dims)
#
# percuda_dims holds (size_x, size_y, size_z) followed by three dimensions per
# argument. Arrays are flattened in row-major order.

my %RANK = (i => 0, f => 1, d => 2);
my @TYPE_BY_RANK = qw(i f d);
my %CTYPE = (i => 'int', f => 'float', d => 'double');
my @AXES = qw(x y z);

our $KERNEL_NAME = 'percuda_kernel';

sub new ($class, %args) {
    my $self = bless {
        ast       => $args{ast},
        dim       => $args{dim},
        arguments => $args{arguments},    # [ [ name, type ], ... ]
        symbols   => [],
        next_id   => 0,
    }, $class;
    Carp::croak("PerCUDA: dim must be 1, 2 or 3") unless $self->{dim} =~ /\A[123]\z/;
    return $self;
}

sub _type_max (@types) { $TYPE_BY_RANK[ max map { $RANK{$_} } @types ] }

sub _error ($msg, $node = undef) {
    my $where = $node && $node->{line} ? " at kernel line $node->{line}" : '';
    Carp::croak("PerCUDA: $msg$where");
}

# Returns { c => $c_source, args => [ { name, type, kind => array|scalar|unused, rank } ] }
sub generate ($self) {
    $self->_analyze_arguments;
    $self->_resolve;
    $self->_infer;
    return { c => $self->_emit_kernel, args => $self->{args} };
}

# ---- argument kinds (array or scalar) are decided by how the kernel uses them ----

sub _analyze_arguments ($self) {
    my (%rank, %scalar);
    _walk($self->{ast}, sub ($n) {
        if ($n->{t} eq 'elem') {
            my $r = scalar @{ $n->{idx} };
            _error("array \$$n->{name} is used with " . "$rank{$n->{name}} and $r subscripts", $n)
                if $rank{ $n->{name} } && $rank{ $n->{name} } != $r;
            $rank{ $n->{name} } = $r;
        }
        elsif ($n->{t} eq 'var') {
            $scalar{ $n->{name} } = 1;
        }
    });

    my %seen;
    my @args;
    for my $pair (@{ $self->{arguments} }) {
        my ($name, $type) = @$pair;
        _error("argument name '$name' is invalid") unless $name =~ /\A[A-Za-z_]\w*\z/;
        _error("argument '$name' is declared twice") if $seen{$name}++;
        _error("argument '$name' has unknown type '$type' (use i, f or d)") unless exists $RANK{$type};
        _error("argument name '$name' is reserved") if $self->_is_builtin_name($name);
        my $kind = $rank{$name} ? 'array' : $scalar{$name} ? 'scalar' : 'unused';
        push @args, { name => $name, type => $type, kind => $kind, rank => $rank{$name} // 0 };
    }
    for my $name (sort keys %rank) {
        _error("array \$$name is not an argument (kernels cannot declare arrays)") unless $seen{$name};
    }
    $self->{args} = \@args;
}

sub _is_builtin_name ($self, $name) {
    for my $i (0 .. $self->{dim} - 1) {
        return 1 if $name eq $AXES[$i] || $name eq "size_$AXES[$i]";
    }
    return 0;
}

sub _walk ($node, $cb) {
    return unless ref $node;
    if (ref $node eq 'ARRAY') {
        _walk($_, $cb) for @$node;
        return;
    }
    $cb->($node) if exists $node->{t};
    for my $key (sort keys %$node) {
        next if $key eq 'sym';
        _walk($node->{$key}, $cb) if ref $node->{$key};
    }
}

# ---- scope resolution ----

sub _new_symbol ($self, %args) {
    my $sym = { type => 'i', %args };
    push @{ $self->{symbols} }, $sym;
    return $sym;
}

sub _resolve ($self) {
    my %global;
    for my $i (0 .. $self->{dim} - 1) {
        my $axis = $AXES[$i];
        $global{$axis} = $self->_new_symbol(name => $axis, cname => "percuda_$axis", fixed => 1);
        $global{"size_$axis"} = $self->_new_symbol(name => "size_$axis", cname => "percuda_size_$axis", fixed => 1);
    }
    for my $arg (@{ $self->{args} }) {
        next if $arg->{kind} eq 'unused';
        $global{ $arg->{name} } = $self->_new_symbol(
            name  => $arg->{name},
            cname => "v_$arg->{name}",
            type  => $arg->{type},
            fixed => 1,
            array => $arg->{kind} eq 'array' ? $arg : undef,
        );
    }
    $self->{scopes} = [ \%global ];
    $self->{loops}  = [];
    $self->_resolve_block($self->{ast});
}

sub _lookup ($self, $name, $node) {
    for my $scope (reverse @{ $self->{scopes} }) {
        return $scope->{$name} if $scope->{$name};
    }
    _error("variable \$$name is not declared (declare it with my)", $node);
}

sub _declare ($self, $node) {
    my $id  = $self->{next_id}++;
    my $sym = $self->_new_symbol(name => $node->{name}, cname => "l${id}_$node->{name}");
    $self->{scopes}[-1]{ $node->{name} } = $sym;
    $node->{sym} = $sym;
    return $sym;
}

sub _resolve_block ($self, $block, $bare = 0) {
    push @{ $self->{scopes} }, {};
    push @{ $self->{loops} }, 'bare' if $bare;
    $self->_resolve_stmt($_) for @{ $block->{body} };
    pop @{ $self->{loops} } if $bare;
    pop @{ $self->{scopes} };
}

sub _resolve_stmt ($self, $s) {
    my $t = $s->{t};
    if ($t eq 'my') {
        $self->_resolve_expr($s->{init}) if $s->{init};
        $self->_declare($s);
    }
    elsif ($t eq 'expr') {
        $self->_resolve_expr($s->{e});
    }
    elsif ($t eq 'block') {
        $self->_resolve_block($s, 1);
    }
    elsif ($t eq 'if') {
        for my $c (@{ $s->{conds} }) {
            $self->_resolve_expr($c->[0]);
            $self->_resolve_block($c->[1]);
        }
        $self->_resolve_block($s->{else}) if $s->{else};
    }
    elsif ($t eq 'while') {
        $self->_resolve_expr($s->{cond});
        $self->_resolve_loop_body($s->{body});
    }
    elsif ($t eq 'for') {
        push @{ $self->{scopes} }, {};
        $self->_resolve_stmt($s->{init}) if $s->{init};
        $self->_resolve_expr($s->{cond}) if $s->{cond};
        $self->_resolve_expr($s->{step}) if $s->{step};
        $self->_resolve_loop_body($s->{body});
        pop @{ $self->{scopes} };
    }
    elsif ($t eq 'foreach') {
        $self->_resolve_expr($s->{from});
        $self->_resolve_expr($s->{to});
        push @{ $self->{scopes} }, {};
        $self->_declare($s);
        $self->_resolve_loop_body($s->{body});
        pop @{ $self->{scopes} };
    }
    elsif ($t eq 'last' || $t eq 'next') {
        my $loop = $self->{loops}[-1];
        _error("'$t' outside of a loop is not supported", $s) unless $loop;
        _error("'$t' inside a bare block is not supported", $s) if $loop eq 'bare';
    }
    elsif ($t eq 'return') {
        # ends the current thread
    }
    else {
        _error("unsupported statement '$t'", $s);
    }
}

sub _resolve_loop_body ($self, $body) {
    push @{ $self->{loops} }, 'loop';
    $self->_resolve_block($body);
    pop @{ $self->{loops} };
}

sub _resolve_expr ($self, $e) {
    my $t = $e->{t};
    if ($t eq 'var') {
        my $sym = $self->_lookup($e->{name}, $e);
        _error("array \$$e->{name} must be used with subscripts", $e) if $sym->{array};
        $e->{sym} = $sym;
    }
    elsif ($t eq 'elem') {
        my $sym = $self->_lookup($e->{name}, $e);
        _error("\$$e->{name} is not an array argument", $e) unless $sym->{array};
        $e->{sym} = $sym;
        $self->_resolve_expr($_) for @{ $e->{idx} };
    }
    elsif ($t eq 'num') { }
    elsif ($t eq 'bin') {
        $self->_resolve_expr($e->{l});
        $self->_resolve_expr($e->{r});
    }
    elsif ($t eq 'un' || $t eq 'pre' || $t eq 'post') {
        $self->_resolve_expr($e->{e});
    }
    elsif ($t eq 'assign') {
        $self->_resolve_expr($e->{rhs});
        $self->_resolve_expr($e->{lhs});
    }
    elsif ($t eq 'cond') {
        $self->_resolve_expr($e->{$_}) for qw(c a b);
    }
    elsif ($t eq 'call') {
        $self->_resolve_expr($_) for @{ $e->{args} };
    }
    else {
        _error("unsupported expression '$t'");
    }
}

# ---- type inference: a local is int unless a float/double value is ever assigned to it ----

sub _infer ($self) {
    my @assigns;
    _walk($self->{ast}, sub ($n) {
        if ($n->{t} eq 'my' && $n->{init}) {
            push @assigns, [ $n->{sym}, $n->{init} ];
        }
        elsif ($n->{t} eq 'assign' && $n->{lhs}{t} eq 'var') {
            my $op  = $n->{op} =~ s/=\z//r;
            my $rhs = $op eq '' || $op eq '&&' || $op eq '||'
                ? $n->{rhs}
                : { t => 'bin', op => $op, l => $n->{lhs}, r => $n->{rhs} };
            push @assigns, [ $n->{lhs}{sym}, $rhs ];
        }
    });
    my $changed = 1;
    while ($changed) {
        $changed = 0;
        for my $a (@assigns) {
            my ($sym, $rhs) = @$a;
            next if $sym->{fixed};
            my $type = _type_max($sym->{type}, $self->_type_of($rhs));
            if ($type ne $sym->{type}) {
                $sym->{type} = $type;
                $changed = 1;
            }
        }
    }
}

sub _type_of ($self, $e) {
    my $t = $e->{t};
    return $e->{float} ? 'f' : 'i' if $t eq 'num';
    return $e->{sym}{type} if $t eq 'var' || $t eq 'elem';
    return $self->_type_of($e->{lhs}) if $t eq 'assign';
    return $self->_type_of($e->{e}) if $t eq 'pre' || $t eq 'post';
    return _type_max($self->_type_of($e->{a}), $self->_type_of($e->{b})) if $t eq 'cond';
    if ($t eq 'un') {
        return $e->{op} eq '-' ? $self->_type_of($e->{e}) : 'i';
    }
    if ($t eq 'call') {
        my $arg = $self->_type_of($e->{args}[0]);
        return $e->{name} eq 'sqrt' ? _type_max('f', $arg)
             : $e->{name} eq 'abs'  ? $arg
             :                        'i';
    }
    if ($t eq 'bin') {
        my ($l, $r) = ($self->_type_of($e->{l}), $self->_type_of($e->{r}));
        my $op = $e->{op};
        return _type_max($l, $r) if $op eq '+' || $op eq '-' || $op eq '*';
        return _type_max('f', $l, $r) if $op eq '/';
        if ($op eq '**') {
            return $l if _nonneg_int_literal($e->{r});
            return _type_max('f', $l);
        }
        return _type_max($l, $r) if ($op eq '||' || $op eq '&&') && _pure($e);
        return 'i';
    }
    _error("cannot infer type of '$t'");
}

sub _nonneg_int_literal ($e) { $e->{t} eq 'num' && !$e->{float} && $e->{value} >= 0 }

sub _pure ($e) {
    my $pure = 1;
    _walk($e, sub ($n) { $pure = 0 if $n->{t} eq 'assign' || $n->{t} eq 'pre' || $n->{t} eq 'post' });
    return $pure;
}

# ---- C emission ----

my $PRELUDE = <<'C';
static inline long long percuda_mod(long long a, long long b) {
    if (b == 0) return 0;
    long long r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
static inline int percuda_powi_i(int x, int n) {
    int r = 1;
    while (n > 0) { if (n & 1) r *= x; x *= x; n >>= 1; }
    return r;
}
static inline float percuda_powi_f(float x, int n) {
    float r = 1.0f;
    unsigned m = n < 0 ? -(unsigned)n : (unsigned)n;
    while (m) { if (m & 1) r *= x; x *= x; m >>= 1; }
    return n < 0 ? 1.0f / r : r;
}
static inline double percuda_powi_d(double x, int n) {
    double r = 1.0;
    unsigned m = n < 0 ? -(unsigned)n : (unsigned)n;
    while (m) { if (m & 1) r *= x; x *= x; m >>= 1; }
    return n < 0 ? 1.0 / r : r;
}
static inline int percuda_abs_i(int x) { return x < 0 ? -x : x; }
C

sub _emit_kernel ($self) {
    my @params;
    my @prologue;
    my $dim = $self->{dim};

    for my $i (0 .. $dim - 1) {
        my $axis = $AXES[$i];
        my ($b, $t, $g) = ("ctaid_$axis", "tid_$axis", "ntid_$axis");
        push @prologue,
            "int percuda_size_$axis = percuda_dims[$i];",
            "int percuda_$axis = __nvvm_read_ptx_sreg_$b() * __nvvm_read_ptx_sreg_$g() + __nvvm_read_ptx_sreg_$t();";
    }
    push @prologue, 'if (' . join(' || ', map { "percuda_$AXES[$_] >= percuda_size_$AXES[$_]" } 0 .. $dim - 1) . ') return;';

    my $k = 0;
    for my $arg (@{ $self->{args} }) {
        my $ctype = $CTYPE{ $arg->{type} };
        if ($arg->{kind} eq 'array') {
            push @params, "$ctype *v_$arg->{name}";
            for my $d (1 .. $arg->{rank} - 1) {
                my $idx = 3 + $k * 3 + $d;
                push @prologue, "int d_$arg->{name}_$d = percuda_dims[$idx];";
            }
        }
        elsif ($arg->{kind} eq 'scalar') {
            push @params, "$ctype *p_$arg->{name}";
            push @prologue, "$ctype v_$arg->{name} = *p_$arg->{name};";
        }
        else {
            push @params, "void *unused_$arg->{name}";
        }
        $k++;
    }
    push @params, 'int *percuda_dims';

    my $body = join '', map { $self->_emit_stmt($_, 1) } @{ $self->{ast}{body} };
    my $prologue = join '', map { "    $_\n" } @prologue;
    my $params = join ', ', @params;
    return "$PRELUDE\n__attribute__((nvptx_kernel)) void $KERNEL_NAME($params) {\n$prologue$body}\n";
}

sub _ind ($level) { '    ' x $level }

sub _emit_block ($self, $block, $level) {
    my $inner = join '', map { $self->_emit_stmt($_, $level + 1) } @{ $block->{body} };
    return "{\n$inner" . _ind($level) . "}";
}

sub _emit_stmt ($self, $s, $level) {
    my $ind = _ind($level);
    my $t   = $s->{t};
    if ($t eq 'my') {
        my $type = $s->{sym}{type};
        my $init = $s->{init} ? $self->_cast($type, $s->{init}) : '0';
        return "$ind$CTYPE{$type} $s->{sym}{cname} = $init;\n";
    }
    if ($t eq 'expr') {
        return $ind . $self->_expr($s->{e}) . ";\n";
    }
    if ($t eq 'block') {
        return $ind . $self->_emit_block($s, $level) . "\n";
    }
    if ($t eq 'if') {
        my @parts;
        for my $c (@{ $s->{conds} }) {
            push @parts, 'if (' . $self->_expr($c->[0]) . ') ' . $self->_emit_block($c->[1], $level);
        }
        my $out = $ind . join(' else ', @parts);
        $out .= ' else ' . $self->_emit_block($s->{else}, $level) if $s->{else};
        return "$out\n";
    }
    if ($t eq 'while') {
        return $ind . 'while (' . $self->_expr($s->{cond}) . ') ' . $self->_emit_block($s->{body}, $level) . "\n";
    }
    if ($t eq 'for') {
        my $init = $s->{init} ? $self->_emit_stmt($s->{init}, $level + 1) : '';
        my $cond = $s->{cond} ? $self->_expr($s->{cond}) : '1';
        my $step = $s->{step} ? $self->_expr($s->{step}) : '';
        return "$ind\{\n$init" . _ind($level + 1) . "for (; $cond; $step) "
            . $self->_emit_block($s->{body}, $level + 1) . "\n$ind}\n";
    }
    if ($t eq 'foreach') {
        # Perl evaluates the range once and the loop variable is a copy per iteration
        my $id = $self->{next_id}++;
        my ($lo, $hi, $it) = ("percuda_lo$id", "percuda_hi$id", "percuda_it$id");
        my $i1 = _ind($level + 1);
        my $i2 = _ind($level + 2);
        return "$ind\{\n"
            . "${i1}int $lo = " . $self->_cast('i', $s->{from}) . ";\n"
            . "${i1}int $hi = " . $self->_cast('i', $s->{to}) . ";\n"
            . "${i1}for (int $it = $lo; $it <= $hi; $it++) {\n"
            . "${i2}$CTYPE{$s->{sym}{type}} $s->{sym}{cname} = $it;\n"
            . "$i2" . $self->_emit_block($s->{body}, $level + 2) . "\n"
            . "$i1}\n$ind}\n";
    }
    return "${ind}break;\n"    if $t eq 'last';
    return "${ind}continue;\n" if $t eq 'next';
    return "${ind}return;\n"   if $t eq 'return';
    _error("unsupported statement '$t'", $s);
}

sub _cast ($self, $type, $e) {
    my $c = $self->_expr($e);
    return $self->_type_of($e) eq $type ? $c : "(($CTYPE{$type})($c))";
}

sub _num ($e) {
    return sprintf('%d', $e->{value}) unless $e->{float};
    my $s = sprintf('%.9g', $e->{value});
    $s .= '.0' unless $s =~ /[.e]/;
    return $s . 'f';
}

sub _expr ($self, $e) {
    my $t = $e->{t};
    return _num($e) if $t eq 'num';
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
        my $type = $self->_type_of($e);
        return '(' . $self->_expr($e->{c}) . ' ? ' . $self->_cast($type, $e->{a}) . ' : ' . $self->_cast($type, $e->{b}) . ')';
    }
    return $self->_assign($e) if $t eq 'assign';
    return $self->_call($e) if $t eq 'call';
    return $self->_bin($e) if $t eq 'bin';
    _error("unsupported expression '$t'");
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
    my $type = $self->_type_of($e->{lhs});
    my $op   = $e->{op};
    return "($lhs = " . $self->_cast($type, $e->{rhs}) . ')' if $op eq '=';
    return "($lhs ? $lhs : ($lhs = " . $self->_cast($type, $e->{rhs}) . '))' if $op eq '||=';
    return "($lhs ? ($lhs = " . $self->_cast($type, $e->{rhs}) . ") : $lhs)" if $op eq '&&=';
    my $bin = { t => 'bin', op => ($op =~ s/=\z//r), l => $e->{lhs}, r => $e->{rhs} };
    return "($lhs = " . $self->_cast($type, $bin) . ')';
}

sub _call ($self, $e) {
    my $arg  = $e->{args}[0];
    my $type = $self->_type_of($e);
    if ($e->{name} eq 'sqrt') {
        return ($type eq 'd' ? '__builtin_sqrt(' : '__builtin_sqrtf(') . $self->_cast($type, $arg) . ')';
    }
    if ($e->{name} eq 'abs') {
        my $fn = { i => 'percuda_abs_i', f => '__builtin_fabsf', d => '__builtin_fabs' }->{$type};
        return "$fn(" . $self->_expr($arg) . ')';
    }
    return $self->_cast('i', $arg) if $e->{name} eq 'int';
    _error("unsupported function '$e->{name}'");
}

sub _bin ($self, $e) {
    my $op   = $e->{op};
    my $type = $self->_type_of($e);
    my ($l, $r) = ($e->{l}, $e->{r});

    if ($op eq '/') {
        return '(' . $self->_cast($type, $l) . ' / ' . $self->_cast($type, $r) . ')';
    }
    if ($op eq '%') {
        return '((int)percuda_mod((long long)(' . $self->_expr($l) . '), (long long)(' . $self->_expr($r) . ')))';
    }
    if ($op eq '**') {
        _error("'**' needs an integer exponent") if $self->_type_of($r) ne 'i';
        return "percuda_powi_$type(" . $self->_cast($type, $l) . ', ' . $self->_expr($r) . ')';
    }
    if ($op eq '&' || $op eq '|' || $op eq '^' || $op eq '<<' || $op eq '>>') {
        return '(' . $self->_cast('i', $l) . " $op " . $self->_cast('i', $r) . ')';
    }
    if (($op eq '||' || $op eq '&&') && _pure($e)) {
        # Perl returns the deciding operand, not a boolean
        my ($lc, $lt, $rt) = ($self->_expr($l), $self->_cast($type, $l), $self->_cast($type, $r));
        return $op eq '||' ? "($lc ? $lt : $rt)" : "($lc ? $rt : $lt)";
    }
    return '(' . $self->_expr($l) . " $op " . $self->_expr($r) . ')';
}

1;
