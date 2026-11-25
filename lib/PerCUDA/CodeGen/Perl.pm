package PerCUDA::CodeGen::Perl;
use v5.40;
use Scalar::Util qw(refaddr);
use PerCUDA::Analyzer qw(is_pure assigned_symbols walk error);

# Emits the kernel as a Perl function for the perl engine. It loops over every
# thread index (Fig. 3(b) of the paper) and follows the GPU semantics where
# Perl differs: comparisons yield 0/1, bit operations work on 32-bit signed
# integers and stores into int variables truncate.

my %BITOP = ('&' => 'band', '|' => 'bor', '^' => 'bxor', '<<' => 'shl', '>>' => 'shr');
my %CMP   = map { $_ => 1 } qw(== != < > <= >=);

sub new ($class, $analyzer) { bless { an => $analyzer }, $class }

sub generate ($self) {
    my $an     = $self->{an};
    my @axes   = $an->axes;
    my @params = map {
        $_->{kind} eq 'array' ? "\$v_$_->{name}" : $_->{kind} eq 'scalar' ? "\$p_$_->{name}" : "\$unused_$_->{name}"
    } @{ $an->args };
    push @params, map { "\$percuda_size_$_" } @axes;

    my $out = "sub {\n    my (" . join(', ', @params) . ") = \@_;\n";
    for my $arg (grep { $_->{kind} eq 'scalar' } @{ $an->args }) {
        my $value = "\$p_$arg->{name}";
        $value = "int($value)" if $arg->{type} eq 'i';
        $out .= "    my \$v_$arg->{name} = $value;\n";
    }
    my $level = 1;
    for my $i (0 .. $#axes) {
        my $label = $i == $#axes ? 'PERCUDA_THREAD: ' : '';
        $out .= _ind($level++) . "${label}for my \$percuda_$axes[$i] (0 .. \$percuda_size_$axes[$i] - 1) {\n";
    }
    $out .= join '', map { $self->_stmt($_, $level) } @{ $an->ast->{body} };
    $out .= _ind(--$level) . "}\n" for @axes;
    return $out . "    return;\n}\n";
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
        my $init = $s->{init} ? $self->_cast($s->{sym}{type}, $s->{init}) : '0';
        return "${ind}my \$$s->{sym}{cname} = $init;\n";
    }
    return $ind . $self->_expr($s->{e}) . ";\n" if $t eq 'expr';
    return $ind . $self->_block($s, $level) . "\n" if $t eq 'block';
    if ($t eq 'if') {
        my @parts = map { '(' . $self->_bool($_->[0]) . ') ' . $self->_block($_->[1], $level) } @{ $s->{conds} };
        my $out = "${ind}if " . join(' elsif ', @parts);
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
        my $cond = $s->{cond} ? $self->_bool($s->{cond}) : '';
        my $step = $s->{step} ? $self->_expr($s->{step}) : '';
        return "$ind\{\n$init" . _ind($level + 1) . "for (; $cond; $step) "
            . $self->_block($s->{body}, $level + 1) . "\n$ind}\n";
    }
    if ($t eq 'foreach') {
        return "${ind}for my \$$s->{sym}{cname} (" . $self->_expr($s->{from}) . ' .. ' . $self->_expr($s->{to}) . ') '
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
    return undef unless $init && $init->{t} eq 'my' && $init->{init} && $cond && $step;
    my $sym = $init->{sym};
    my $an  = $self->{an};
    return undef unless $sym->{type} eq 'i';
    return undef unless $cond->{t} eq 'bin' && ($cond->{op} eq '<' || $cond->{op} eq '<=');
    return undef unless $cond->{l}{t} eq 'var' && $cond->{l}{sym} == $sym;
    my $is_inc = ($step->{t} eq 'pre' || $step->{t} eq 'post') && $step->{op} eq '++'
        && $step->{e}{t} eq 'var' && $step->{e}{sym} == $sym;
    return undef unless $is_inc;

    my $limit = $cond->{r};
    return undef unless $an->type_of($limit) eq 'i' && $an->type_of($init->{init}) eq 'i';
    my $simple = 1;
    my @vars;
    walk($limit, sub ($n) {
        if    ($n->{t} eq 'var') { push @vars, $n->{sym} }
        elsif ($n->{t} ne 'num' && !($n->{t} eq 'bin' && $n->{op} =~ /\A[-+*]\z/)) { $simple = 0 }
    });
    return undef unless $simple;
    my $assigned = assigned_symbols($s->{body});
    return undef if grep { $assigned->{ refaddr $_ } } $sym, @vars;

    my $hi = $self->_expr($limit);
    $hi = "$hi - 1" if $cond->{op} eq '<';
    return $self->_expr($init->{init}) . " .. $hi";
}

sub _cast ($self, $type, $e) {
    my $code = $self->_expr($e);
    return $type eq 'i' && $self->{an}->type_of($e) ne 'i' ? "int($code)" : $code;
}

sub _num ($e) {
    return sprintf('%d', $e->{value}) unless $e->{float};
    my $s = sprintf('%.17g', $e->{value});
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
    my $t = $e->{t};
    return _num($e) if $t eq 'num';
    return "\$$e->{sym}{cname}" if $t eq 'var';
    if ($t eq 'elem') {
        return "\$v_$e->{name}" . join('', map { '->[' . $self->_expr($_) . ']' } @{ $e->{idx} });
    }
    if ($t eq 'un') {
        my $inner = $self->_expr($e->{e});
        return "(-$inner)" if $e->{op} eq '-';
        return '(' . $self->_bool($e->{e}) . ' ? 0 : 1)' if $e->{op} eq '!';
        return "PerCUDA::Runtime::bnot($inner)";
    }
    return "($e->{op}" . $self->_expr($e->{e}) . ')' if $t eq 'pre';
    return '(' . $self->_expr($e->{e}) . "$e->{op})" if $t eq 'post';
    if ($t eq 'cond') {
        return '(' . $self->_bool($e->{c}) . ' ? ' . $self->_expr($e->{a}) . ' : ' . $self->_expr($e->{b}) . ')';
    }
    if ($t eq 'assign') {
        my $lhs  = $self->_expr($e->{lhs});
        my $type = $self->{an}->type_of($e->{lhs});
        my $op   = $e->{op};
        return "($lhs $op " . $self->_cast($type, $e->{rhs}) . ')' if $op eq '=' || $op eq '||=' || $op eq '&&=';
        if (($op eq '+=' || $op eq '-=' || $op eq '*=') && ($type ne 'i' || $self->{an}->type_of($e->{rhs}) eq 'i')) {
            return "($lhs $op " . $self->_expr($e->{rhs}) . ')';
        }
        my $bin = { t => 'bin', op => ($op =~ s/=\z//r), l => $e->{lhs}, r => $e->{rhs} };
        return "($lhs = " . $self->_cast($type, $bin) . ')';
    }
    return "$e->{name}(" . $self->_expr($e->{args}[0]) . ')' if $t eq 'call';
    if ($t eq 'bin') {
        my $op = $e->{op};
        return '(' . $self->_bool($e) . ' ? 1 : 0)' if $CMP{$op};
        return '(' . $self->_bool($e) . ' ? 1 : 0)' if ($op eq '||' || $op eq '&&') && !is_pure($e);
        my ($l, $r) = ($self->_expr($e->{l}), $self->_expr($e->{r}));
        return "PerCUDA::Runtime::$BITOP{$op}($l, $r)" if $BITOP{$op};
        return "($l $op $r)";
    }
    error("unsupported expression '$t'");
}

1;
