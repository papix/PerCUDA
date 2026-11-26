package PerCUDA::CodeGen::Perl;
use v5.40;
use Scalar::Util      qw(refaddr);
use PerCUDA::Analyzer qw(is_pure assigned_symbols walk error);

# Emits the kernel as a Perl function for the perl engine. It loops over every
# thread index (Fig. 3(b) of the paper) and follows the GPU semantics where
# Perl differs: single precision values are rounded after every operation,
# integers wrap around at 32 bits, division by zero gives inf/nan, comparisons
# yield 0/1, bit operations work on 32-bit signed integers, and conversions
# truncate or round like C casts.

my %BITOP = ('&' => 'band', '|' => 'bor', '^' => 'bxor', '<<' => 'shl', '>>' => 'shr');
my %CMP   = map { $_ => 1 } qw(== != < > <= >=);

# mode: 'compat' (default) follows the GPU semantics, with Perl's own exp,
# log, sin, cos, atan2 and pow; 'exact' also takes those from PerCUDA::Math,
# as the GPU does in that mode; 'fast' skips single precision rounding and
# 32-bit integer wrapping, which cost the most.
sub new ($class, $analyzer, %opt) {
    my $mode = $opt{mode} // 'compat';
    return bless { an => $analyzer, exact => $mode ne 'fast', portable => $mode eq 'exact', check => $opt{check} },
        $class;
}

sub generate ($self) {
    my $an     = $self->{an};
    my @axes   = $an->axes;
    my @params = map {
        $_->{kind} eq 'array' ? "\$v_$_->{name}" : $_->{kind} eq 'scalar' ? "\$p_$_->{name}" : "\$unused_$_->{name}"
    } @{ $an->args };
    push @params, map { "\$percuda_size_$_" } @axes;
    push @params, '$percuda_xs';    # optional: the x indices to run (array reference)

    my $out = "sub {\n    my (" . join(', ', @params) . ") = \@_;\n";
    for my $arg (grep { $_->{kind} eq 'scalar' } @{ $an->args }) {
        my $value = "\$p_$arg->{name}";
        $value = "int($value)"              if $arg->{type} eq 'i';
        $value = $self->_round('f', $value) if $arg->{type} eq 'f';
        $out .= "    my \$v_$arg->{name} = $value;\n";
    }
    my $level = 1;
    for my $i (0 .. $#axes) {
        my $label = $i == $#axes ? 'PERCUDA_THREAD: ' : '';
        my $range = "0 .. \$percuda_size_$axes[$i] - 1";
        $range = "\$percuda_xs ? \@\$percuda_xs : ($range)" if $i == 0;
        $out .= _ind($level++) . "${label}for my \$percuda_$axes[$i] ($range) {\n";
    }
    $out .= join '', map { $self->_stmt($_, $level) } @{ $an->ast->{body} };
    $out .= _ind(--$level) . "}\n" for @axes;
    return $out . "    return;\n}\n";
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
        my $init = $s->{init} ? $self->_cast($s->{sym}{type}, $s->{init}) : '0';
        return "${ind}my \$$s->{sym}{cname} = $init;\n";
    }
    return $ind . $self->_void($s->{e}) . ";\n"    if $t eq 'expr';
    return $ind . $self->_block($s, $level) . "\n" if $t eq 'block';
    if ($t eq 'if') {
        my @parts = map { '(' . $self->_bool($_->[0]) . ') ' . $self->_block($_->[1], $level) } @{ $s->{conds} };
        my $out   = "${ind}if " . join(' elsif ', @parts);
        $out .= ' else ' . $self->_block($s->{else}, $level) if $s->{else};
        return "$out\n";
    }
    if ($t eq 'while') {
        return "${ind}while (" . $self->_bool($s->{cond}) . ') ' . $self->_block($s->{body}, $level) . "\n";
    }
    if ($t eq 'for') {
        if (my $range = $self->_counting_loop($s)) {
            return "${ind}for my \$$s->{init}{sym}{cname} ($range) " . $self->_block($s->{body}, $level) . "\n";
        }
        my $init = $s->{init} ? $self->_stmt($s->{init}, $level + 1) : '';
        my $cond = $s->{cond} ? $self->_bool($s->{cond})             : '';
        my $step = $s->{step} ? $self->_void($s->{step})             : '';
        return
              "$ind\{\n$init"
            . _ind($level + 1)
            . "for (; $cond; $step) "
            . $self->_block($s->{body}, $level + 1)
            . "\n$ind}\n";
    }
    if ($t eq 'foreach') {
        return
              "${ind}for my \$$s->{sym}{cname} ("
            . $self->_expr($s->{from}) . ' .. '
            . $self->_expr($s->{to}) . ') '
            . $self->_block($s->{body}, $level) . "\n";
    }
    return "${ind}last;\n"                if $t eq 'last';
    return "${ind}next;\n"                if $t eq 'next';
    return "${ind}next PERCUDA_THREAD;\n" if $t eq 'return';
    error("unsupported statement '$t'", $s);
}

# for (my $i = A; $i < B; $i++) runs much faster in perl as for my $i (A .. B - 1).
# This is only done when it cannot change the result: everything is int, the
# body does not assign $i, and B is made of variables the loop never assigns.
sub _counting_loop ($self, $s) {
    my ($init, $cond, $step) = @$s{qw(init cond step)};
    return unless $init && $init->{t} eq 'my' && $init->{init} && $cond && $step;
    my $sym = $init->{sym};
    my $an  = $self->{an};
    return unless $sym->{type} eq 'i';
    return unless $cond->{t} eq 'bin'    && ($cond->{op} eq '<' || $cond->{op} eq '<=');
    return unless $cond->{l}{t} eq 'var' && $cond->{l}{sym} == $sym;
    my $is_inc =
           ($step->{t} eq 'pre' || $step->{t} eq 'post')
        && $step->{op} eq '++'
        && $step->{e}{t} eq 'var'
        && $step->{e}{sym} == $sym;
    return unless $is_inc;

    my $limit = $cond->{r};
    return unless $an->type_of($limit) eq 'i' && $an->type_of($init->{init}) eq 'i';
    my $simple = 1;
    my @vars;
    walk(
        $limit,
        sub ($n) {
            if    ($n->{t} eq 'var')                                                   { push @vars, $n->{sym} }
            elsif ($n->{t} ne 'num' && !($n->{t} eq 'bin' && $n->{op} =~ /\A[-+*]\z/)) { $simple = 0 }
        }
    );
    return unless $simple;
    my $assigned = assigned_symbols($s->{body});
    return if grep { $assigned->{ refaddr $_ } } $sym, @vars;

    my $hi = $self->_expr($limit);
    $hi = "$hi - 1" if $cond->{op} eq '<';
    return $self->_expr($init->{init}) . " .. $hi";
}

# An expression whose value is discarded: $i++ needs no copy of the old value.
sub _void ($self, $e) {
    return $self->_expr($e->{t} eq 'post' ? { %$e, t => 'pre' } : $e);
}

# Converts like a C cast: truncation to int, rounding to single precision.
sub _cast ($self, $type, $e) {
    my $code = $self->_expr($e);
    my $from = $self->{an}->type_of($e);
    return $code        if $from eq $type;
    return "int($code)" if $type eq 'i';
    return $self->_round($type, $code);
}

sub _num ($self, $e) {
    return sprintf('%d', $e->{value}) unless $e->{float};
    my $value = $self->{exact} && $self->{an}->real eq 'f' ? unpack('f', pack('f', $e->{value})) : $e->{value};
    my $s     = sprintf('%.17g', $value);
    return $s =~ /[.e]/ ? $s : "$s.0";
}

# An expression whose value is only tested for truth needs no 0/1 normalization.
sub _bool ($self, $e) {
    my $t = $e->{t};
    if ($t eq 'bin' && $CMP{ $e->{op} }) {
        return '(' . $self->_expr($e->{l}) . " $e->{op} " . $self->_expr($e->{r}) . ')';
    }
    if ($t eq 'bin' && ($e->{op} eq '&&' || $e->{op} eq '||')) {
        return '(' . $self->_bool($e->{l}) . " $e->{op} " . $self->_bool($e->{r}) . ')';
    }
    return '(!' . $self->_bool($e->{e}) . ')' if $t eq 'un' && $e->{op} eq '!';
    return $self->_expr($e);
}

sub _expr ($self, $e) {
    my $an = $self->{an};
    my $t  = $e->{t};
    return $self->_num($e)      if $t eq 'num';
    return "\$$e->{sym}{cname}" if $t eq 'var';
    if ($t eq 'elem') {
        if ($self->{check}) {    # a reference to the element, after checking every index
            my @names  = @{ $an->thread_names };
            my @axes   = $an->axes;
            my $thread = join ', ',
                map { (defined $names[$_] ? "'$names[$_]'" : 'undef') . ", \$percuda_$axes[$_]" } 0 .. $#axes;
            return
                  "\${ PerCUDA::Runtime::at(\$v_$e->{name}, '$e->{name}', "
                . ($e->{line} // 0)
                . ", [$thread], "
                . join(', ', map { $self->_expr($_) } @{ $e->{idx} }) . ') }';
        }
        return "\$v_$e->{name}" . join('', map { '->[' . $self->_expr($_) . ']' } @{ $e->{idx} });
    }
    if ($t eq 'un') {
        my $inner = $self->_expr($e->{e});
        return $self->_round($an->type_of($e) eq 'i' ? 'i' : 'd', "(-$inner)") if $e->{op} eq '-';
        return '(' . $self->_bool($e->{e}) . ' ? 0 : 1)'                       if $e->{op} eq '!';
        return "PerCUDA::Runtime::bnot($inner)";
    }
    if ($t eq 'pre' || $t eq 'post') {
        my $var  = $self->_expr($e->{e});
        my $type = $an->type_of($e->{e});
        return $t eq 'pre' ? "($e->{op}$var)" : "($var$e->{op})" if $type eq 'd' || !$self->{exact};
        my $op  = substr $e->{op}, 0, 1;
        my $new = $self->_round($type, "$var $op 1");
        return $t eq 'pre' ? "($var = $new)" : "do { my \$percuda_old = $var; $var = $new; \$percuda_old }";
    }
    if ($t eq 'cond') {
        my $type = $an->type_of($e);
        return
              '('
            . $self->_bool($e->{c}) . ' ? '
            . $self->_cast($type, $e->{a}) . ' : '
            . $self->_cast($type, $e->{b}) . ')';
    }
    if ($t eq 'assign') {
        my $lhs  = $self->_expr($e->{lhs});
        my $type = $an->type_of($e->{lhs});
        my $op   = $e->{op};
        return "($lhs $op " . $self->_cast($type, $e->{rhs}) . ')' if $op eq '=' || $op eq '||=' || $op eq '&&=';
        my $rtype  = $an->type_of($e->{rhs});
        my $native = $type eq 'd' || (!$self->{exact} && ($type eq 'f' || $rtype eq 'i'));
        if (($op eq '+=' || $op eq '-=' || $op eq '*=') && $native) {
            return "($lhs $op " . $self->_expr($e->{rhs}) . ')';
        }
        my $bin = { t => 'bin', op => ($op =~ s/=\z//r), l => $e->{lhs}, r => $e->{rhs} };
        return "($lhs = " . $self->_cast($type, $bin) . ')';
    }
    if ($t eq 'call') {
        my $type = $an->type_of($e);
        return 'int(' . $self->_expr($e->{args}[0]) . ')' if $e->{name} eq 'int';
        return 'abs(' . $self->_expr($e->{args}[0]) . ')' if $e->{name} eq 'abs';
        my $name = $e->{name};
        if ($name eq 'random') {    # integer arithmetic only: identical in every mode
            return "PerCUDA::Runtime::random_$type(" . join(', ', map { $self->_cast('i', $_) } @{ $e->{args} }) . ')';
        }
        my $args = join ', ', map { $self->_cast($type, $_) } @{ $e->{args} };
        return "PerCUDA::Math::${name}_$type($args)"
            if $self->{portable} && grep { $_ eq $name } @PerCUDA::Math::FUNCTIONS;

        # Perl's own functions, without the errors the GPU does not raise
        my $code =
              $name eq 'floor' || $name eq 'ceil' ? "POSIX::$name($args)"
            : $name eq 'sqrt'  || $name eq 'log'  ? "PerCUDA::Runtime::safe_$name($args)"
            :                                       "$name($args)";
        return $self->_round($type, $code);
    }
    if ($t eq 'bin') {
        my $op   = $e->{op};
        my $type = $an->type_of($e);
        return '(' . $self->_bool($e) . ' ? 1 : 0)' if $CMP{$op};
        if ($op eq '||' || $op eq '&&') {
            return '(' . $self->_bool($e) . ' ? 1 : 0)' unless is_pure($e);

            # the deciding operand, converted to the type of the expression as on the GPU
            my ($lc, $lt, $rt) = ($self->_expr($e->{l}), $self->_cast($type, $e->{l}), $self->_cast($type, $e->{r}));
            return $op eq '||' ? "($lc ? $lt : $rt)" : "($lc ? $rt : $lt)";
        }
        if ($BITOP{$op}) {
            return "PerCUDA::Runtime::$BITOP{$op}(" . $self->_expr($e->{l}) . ', ' . $self->_expr($e->{r}) . ')';
        }
        if ($op eq '**') {

            # repeated multiplication in the same order as the GPU's percuda_powi_*
            if ($an->type_of($e->{r}) eq 'i') {
                return
                      "PerCUDA::Runtime::powi_$type("
                    . $self->_cast($type, $e->{l}) . ', '
                    . $self->_expr($e->{r}) . ')';
            }
            my ($l, $r) = ($self->_cast($type, $e->{l}), $self->_cast($type, $e->{r}));
            return "PerCUDA::Math::pow_$type($l, $r)" if $self->{portable};
            return $self->_round($type, "($l ** $r)");
        }

        # division and modulus by zero give inf/nan and 0 as on the GPU, instead of dying
        if ($op eq '%') {
            my ($l, $r) = ($self->_expr($e->{l}), $self->_expr($e->{r}));
            return "PerCUDA::Runtime::mod($l, $r)" unless is_pure($e->{r});
            my $test = $an->type_of($e->{r}) eq 'i' ? $r : "int($r)";
            return "($test ? $l % $r : 0)";
        }
        my ($l, $r) = ($self->_cast($type, $e->{l}), $self->_cast($type, $e->{r}));
        if ($op eq '/') {
            my $code =
                is_pure($e->{r})
                ? "($r != 0 ? $l / $r : PerCUDA::Runtime::div0($l, $r))"
                : "PerCUDA::Runtime::div($l, $r)";
            return $self->_round($type, $code);
        }
        return $self->_round($type, "($l $op $r)");
    }
    error("unsupported expression '$t'");
}

# Makes the result of an operation of $type what the GPU computes. In compat
# mode, single precision values are rounded (rounding every float result of
# + - * / and sqrt computed in double gives exactly the single precision
# result) and integers wrap around at 32 bits.
sub _round ($self, $type, $code) {
    return $code unless $self->{exact};
    return "unpack('f', pack('f', $code))"                    if $type eq 'f';
    return "(($code + 2147483648) % 4294967296 - 2147483648)" if $type eq 'i';
    return $code;
}

1;
