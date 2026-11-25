package PerCUDA::Parser;
use v5.40;
use Carp ();

# Parses the Perl subset accepted in PerCUDA kernels into an AST.
#
# Statements: { t => 'my' | 'expr' | 'if' | 'for' | 'foreach' | 'while' | 'block'
#                    | 'last' | 'next' | 'return', ... }
# Expressions: { t => 'num' | 'var' | 'elem' | 'bin' | 'un' | 'assign' | 'pre' | 'post'
#                     | 'cond' | 'call', ... }

my @PUNCT = sort { length $b <=> length $a } split ' ', q{
    **= ||= &&= <<= >>= <=>
    ** ++ -- -> += -= *= /= %= &= |= ^= == != <= >= && || << >> ..
    + - * / % = < > ! ~ & | ^ ( ) [ ] { } ; , ? :
};
my $PUNCT_RE = join '|', map { quotemeta } @PUNCT;

my %KEYWORD = map { $_ => 1 } qw(
    my if elsif else unless for foreach while until last next return and or not
);

our %BUILTIN = (sqrt => 1, abs => 1, int => 1);

sub new ($class, %args) {
    return bless { source => $args{source} // '' }, $class;
}

sub parse ($self) {
    $self->{tokens} = $self->_tokenize($self->{source});
    $self->{pos}    = 0;
    my @stmts;
    push @stmts, $self->_statement until $self->_peek->{type} eq 'eof';
    return { t => 'block', body => \@stmts };
}

sub _tokenize ($self, $src) {
    my @tokens;
    my $line = 1;
    pos($src) = 0;
    while (1) {
        if ($src =~ /\G([ \t\r\f]+|#[^\n]*)/gc) { next }
        if ($src =~ /\G\n/gc)                   { $line++; next }
        last if pos($src) >= length $src;
        if ($src =~ /\G\$([A-Za-z_]\w*)/gc) {
            push @tokens, { type => 'var', value => $1, line => $line };
        }
        elsif ($src =~ /\G(0[xX][0-9A-Fa-f_]+|0[bB][01_]+)/gc) {
            push @tokens, { type => 'num', value => oct($1 =~ tr/_//dr), float => 0, line => $line };
        }
        elsif ($src =~ /\G((?:\d[\d_]*(?:\.(?!\.)\d*)?|\.\d+)(?:[eE][-+]?\d+)?)/gc) {
            my $text = $1 =~ tr/_//dr;
            push @tokens,
                {
                type  => 'num',
                value => $text + 0,
                float => ($text =~ /[.eE]/ ? 1 : 0),
                line  => $line,
                };
        }
        elsif ($src =~ /\G([A-Za-z_]\w*)/gc) {
            push @tokens, { type => ($KEYWORD{$1} ? 'kw' : 'ident'), value => $1, line => $line };
        }
        elsif ($src =~ /\G(?:"|'|`)/gc) {
            Carp::croak("PerCUDA: string literals are not supported in kernels at line $line");
        }
        elsif ($src =~ /\G($PUNCT_RE)/gc) {
            push @tokens, { type => 'op', value => $1, line => $line };
        }
        else {
            my $ch = substr $src, pos($src), 1;
            Carp::croak("PerCUDA: unsupported character '$ch' in kernel at line $line");
        }
    }
    push @tokens, { type => 'eof', value => '', line => $line };
    return \@tokens;
}

sub _peek ($self, $n = 0) { return $self->{tokens}[ $self->{pos} + $n ] // $self->{tokens}[-1] }
sub _next ($self)         { return $self->{tokens}[ $self->{pos}++ ] }

sub _is ($self, $type, $value = undef, $n = 0) {
    my $tok = $self->_peek($n);
    return $tok->{type} eq $type && (!defined $value || $tok->{value} eq $value);
}

sub _accept ($self, $type, $value = undef) {
    return $self->_is($type, $value) ? $self->_next : undef;
}

sub _expect ($self, $type, $value = undef) {
    my $tok = $self->_accept($type, $value);
    return $tok if $tok;
    $self->_error('expected ' . ($value // $type));
}

sub _error ($self, $msg) {
    my $tok  = $self->_peek;
    my $near = $tok->{type} eq 'eof' ? 'end of kernel' : "'$tok->{value}'";
    Carp::croak("PerCUDA: syntax error at kernel line $tok->{line} near $near: $msg");
}

# ---- statements ----

sub _block ($self) {
    $self->_expect(op => '{');
    my @stmts;
    until ($self->_accept(op => '}')) {
        $self->_error('unterminated block') if $self->_is('eof');
        push @stmts, $self->_statement;
    }
    return { t => 'block', body => \@stmts };
}

sub _statement ($self) {
    return { t => q{block}, body => [] } if $self->_accept(op => q{;});    # empty statement
    my $tok = $self->_peek;

    return $self->_block if $tok->{type} eq 'op' && $tok->{value} eq '{';

    if ($tok->{type} eq 'kw') {
        my $kw = $tok->{value};
        if ($kw eq 'if' || $kw eq 'unless') {
            $self->_next;
            my @conds;
            my $cond = $self->_paren_expr;
            $cond = { t => 'un', op => '!', e => $cond } if $kw eq 'unless';
            push @conds, [ $cond, $self->_block ];
            while ($self->_accept(kw => 'elsif')) {
                push @conds, [ $self->_paren_expr, $self->_block ];
            }
            my $else = $self->_accept(kw => 'else') ? $self->_block : undef;
            return { t => 'if', conds => \@conds, else => $else };
        }
        if ($kw eq 'while' || $kw eq 'until') {
            $self->_next;
            my $cond = $self->_paren_expr;
            $cond = { t => 'un', op => '!', e => $cond } if $kw eq 'until';
            return { t => 'while', cond => $cond, body => $self->_block };
        }
        if ($kw eq 'for' || $kw eq 'foreach') {
            $self->_next;
            return $self->_for;
        }
    }

    my $stmt = $self->_simple_statement;

    # statement modifiers: STMT if EXPR; / STMT unless EXPR;
    if (my $mod = $self->_accept(kw => 'if') // $self->_accept(kw => 'unless')) {
        my $cond = $self->_expr;
        $cond = { t => 'un', op    => '!', e => $cond } if $mod->{value} eq 'unless';
        $stmt = { t => 'if', conds => [ [ $cond, { t => 'block', body => [$stmt] } ] ], else => undef };
    }
    $self->_expect(op => ';') unless $self->_is(op => '}') || $self->_is('eof');
    return $stmt;
}

sub _simple_statement ($self) {
    return $self->_my if $self->_is(kw => 'my');
    for my $kw (qw(last next return)) {
        if ($self->_accept(kw => $kw)) {
            $self->_error("'$kw' with a value is not supported")
                unless $self->_is(op => ';')
                || $self->_is(op => '}')
                || $self->_is(kw => 'if')
                || $self->_is(kw => 'unless')
                || $self->_is('eof');
            return { t => $kw };
        }
    }
    return { t => 'expr', e => $self->_expr };
}

sub _my ($self) {
    my $tok = $self->_expect(kw => 'my');
    $self->_error('only scalar variables can be declared with my') unless $self->_is('var');
    my $name = $self->_next->{value};
    my $init = $self->_accept(op => '=') ? $self->_assign_expr : undef;
    return { t => 'my', name => $name, init => $init, line => $tok->{line} };
}

sub _for ($self) {

    # foreach style: for my $i (A .. B) { ... }
    if ($self->_is(kw => 'my') && $self->_is(var => undef, 1) && $self->_is(op => '(', 2)) {
        $self->_next;
        my $var = $self->_next;
        $self->_expect(op => '(');
        my $from = $self->_expr_no_range;
        $self->_expect(op => '..');
        my $to = $self->_expr_no_range;
        $self->_expect(op => ')');
        return {
            t    => 'foreach',
            name => $var->{value},
            from => $from,
            to   => $to,
            body => $self->_block,
            line => $var->{line}
        };
    }
    $self->_error('foreach needs a lexical loop variable (for my $i (A .. B))') unless $self->_is(op => '(');

    # C style: for (INIT; COND; STEP) { ... }
    $self->_expect(op => '(');
    my $init = $self->_is(op => ';') ? undef : $self->_simple_statement;
    $self->_expect(op => ';');
    my $cond = $self->_is(op => ';') ? undef : $self->_expr;
    $self->_expect(op => ';');
    my $step = $self->_is(op => ')') ? undef : $self->_expr;
    $self->_expect(op => ')');
    return { t => 'for', init => $init, cond => $cond, step => $step, body => $self->_block };
}

sub _paren_expr ($self) {
    $self->_expect(op => '(');
    my $e = $self->_expr;
    $self->_expect(op => ')');
    return $e;
}

# ---- expressions (Perl precedence, lowest first) ----

sub _expr ($self) { return $self->_low_or }

sub _low_or ($self) {
    my $l = $self->_low_and;
    while ($self->_accept(kw => 'or')) {
        $l = { t => 'bin', op => '||', l => $l, r => $self->_low_and };
    }
    return $l;
}

sub _low_and ($self) {
    my $l = $self->_low_not;
    while ($self->_accept(kw => 'and')) {
        $l = { t => 'bin', op => '&&', l => $l, r => $self->_low_not };
    }
    return $l;
}

sub _low_not ($self) {
    return { t => 'un', op => '!', e => $self->_low_not } if $self->_accept(kw => 'not');
    return $self->_assign_expr;
}

my %ASSIGN_OP = map { $_ => 1 } qw(= += -= *= /= %= **= &= |= ^= <<= >>= &&= ||=);

sub _assign_expr ($self) {
    my $l   = $self->_ternary;
    my $tok = $self->_peek;
    if ($tok->{type} eq 'op' && $ASSIGN_OP{ $tok->{value} }) {
        $self->_error('left side of assignment must be a variable or array element')
            unless $l->{t} eq 'var' || $l->{t} eq 'elem';
        $self->_next;
        return { t => 'assign', op => $tok->{value}, lhs => $l, rhs => $self->_assign_expr };
    }
    return $l;
}

sub _ternary ($self) {
    my $c = $self->_range;
    if ($self->_accept(op => '?')) {
        my $a = $self->_assign_expr;
        $self->_expect(op => ':');
        my $b = $self->_assign_expr;
        return { t => 'cond', c => $c, a => $a, b => $b };
    }
    return $c;
}

sub _range ($self) {
    my $e = $self->_oror;
    $self->_error("range operator '..' is only supported in foreach") if $self->_is(op => '..');
    return $e;
}

sub _expr_no_range ($self) { return $self->_oror }

sub _binary_level ($self, $next, @ops) {
    my $l = $self->$next;
    OUTER: while (1) {
        for my $op (@ops) {
            if ($self->_accept(op => $op)) {
                $l = { t => 'bin', op => $op, l => $l, r => $self->$next };
                next OUTER;
            }
        }
        last;
    }
    return $l;
}

sub _oror   ($self) { return $self->_binary_level('_andand',   '||') }
sub _andand ($self) { return $self->_binary_level('_bitor',    '&&') }
sub _bitor  ($self) { return $self->_binary_level('_bitand',   '|', '^') }
sub _bitand ($self) { return $self->_binary_level('_equality', '&') }

sub _equality ($self) {
    my $e = $self->_binary_level('_relational', '==', '!=', '<=>');
    $self->_error("operator '<=>' is not supported") if _contains_op($e, '<=>');
    return $e;
}

sub _contains_op ($e, $op) { return $e->{t} eq 'bin' && $e->{op} eq $op }

sub _relational     ($self) { return $self->_binary_level('_shift',          '<=', '>=', '<', '>') }
sub _shift          ($self) { return $self->_binary_level('_additive',       '<<', '>>') }
sub _additive       ($self) { return $self->_binary_level('_multiplicative', '+',  '-') }
sub _multiplicative ($self) { return $self->_binary_level('_unary',          '*',  '/', '%') }

sub _unary ($self) {
    for my $op (qw(! ~ - +)) {
        if ($self->_accept(op => $op)) {
            my $e = $self->_unary;
            return $op eq '+' ? $e : { t => 'un', op => $op, e => $e };
        }
    }
    return $self->_power;
}

sub _power ($self) {
    my $base = $self->_incdec;
    if ($self->_accept(op => '**')) {

        # right associative and binds tighter than unary minus on the left only
        return { t => 'bin', op => '**', l => $base, r => $self->_unary };
    }
    return $base;
}

sub _incdec ($self) {
    for my $op (qw(++ --)) {
        if ($self->_accept(op => $op)) {
            my $e = $self->_postfix;
            $self->_error("'$op' needs a variable or array element") unless $e->{t} eq 'var' || $e->{t} eq 'elem';
            return { t => 'pre', op => $op, e => $e };
        }
    }
    my $e = $self->_postfix;
    for my $op (qw(++ --)) {
        if ($self->_accept(op => $op)) {
            $self->_error("'$op' needs a variable or array element") unless $e->{t} eq 'var' || $e->{t} eq 'elem';
            return { t => 'post', op => $op, e => $e };
        }
    }
    return $e;
}

sub _postfix ($self) {
    my $tok = $self->_peek;
    if ($tok->{type} eq 'var') {
        $self->_next;
        my @idx;
        while (1) {
            if ($self->_is(op => '->') && $self->_is(op => '[', 1)) {
                $self->_next;
            }
            elsif (!(@idx && $self->_is(op => '['))) {
                last;
            }
            $self->_expect(op => '[');
            push @idx, $self->_expr;
            $self->_expect(op => ']');
        }
        $self->_error('arrays deeper than 3 dimensions are not supported') if @idx > 3;
        return @idx
            ? { t => 'elem', name => $tok->{value}, idx => \@idx, line => $tok->{line} }
            : { t => 'var', name => $tok->{value}, line => $tok->{line} };
    }
    return $self->_primary;
}

sub _primary ($self) {
    my $tok = $self->_peek;
    if ($tok->{type} eq 'num') {
        $self->_next;
        return { t => 'num', value => $tok->{value}, float => $tok->{float} };
    }
    if ($self->_accept(op => '(')) {
        my $e = $self->_expr;
        $self->_expect(op => ')');
        return $e;
    }
    if ($tok->{type} eq 'ident') {
        $self->_next;
        Carp::croak("PerCUDA: function '$tok->{value}' is not supported in kernels at line $tok->{line}")
            unless $BUILTIN{ $tok->{value} };
        my $arg;
        if ($self->_accept(op => '(')) {
            $arg = $self->_expr;
            $self->_expect(op => ')');
        }
        else {
            $arg = $self->_shift;    # named unary operator: sqrt $x + 1 is sqrt($x + 1)
        }
        return { t => 'call', name => $tok->{value}, args => [$arg] };
    }
    $self->_error('unexpected token');
}

1;
