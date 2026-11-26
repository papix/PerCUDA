package PerCUDA::Analyzer;
use v5.40;
use Carp ();
use Exporter 'import';
use List::Util   qw(max);
use Scalar::Util qw(refaddr);

# Semantic analysis of a kernel AST (PerCUDA::Parser): decides whether each
# argument is an array or a scalar, resolves variables to symbols and infers
# the type of every local. The code generators (PerCUDA::CodeGen::CUDA and
# PerCUDA::CodeGen::Perl) and the engines work from its result.

our @EXPORT_OK = qw(walk is_pure assigned_symbols nonneg_int_literal type_max error);

my %RANK         = (i => 0, f => 1, d => 2);
my @TYPE_BY_RANK = qw(i f d);
our @AXES = qw(x y z);

sub new ($class, %args) {
    my $self = bless {
        ast       => $args{ast},
        dim       => $args{dim},
        arguments => $args{arguments},    # [ [ name, type ], ... ]
        real      => ($args{precision} // 'single') eq 'double' ? 'd' : 'f',
        next_id   => 0,
    }, $class;
    Carp::croak("PerCUDA: dim must be 1, 2 or 3") unless defined $self->{dim} && $self->{dim} =~ /\A[123]\z/;
    $self->_analyze_arguments;
    $self->_resolve;
    $self->_infer;
    $self->_analyze_writes;
    return $self;
}

sub ast  ($self) { return $self->{ast} }
sub dim  ($self) { return $self->{dim} }
sub real ($self) { return $self->{real} }
sub axes ($self) { return @AXES[ 0 .. $self->{dim} - 1 ] }

# [ { name, type, kind => array|scalar|unused, rank, read, written, covered } ] in declaration order.
# covered: the array is never read and every thread stores into its own
# element ($a->[$x][$y] = ... for dim 2) unconditionally, so when its shape is
# the grid, every element is overwritten.
sub args ($self) { return $self->{args} }

# True when every store into an array uses $x as its first index, so threads
# with different $x never write the same element.
sub writes_own_rows ($self) { return $self->{writes_own_rows} }

sub new_id ($self) { return $self->{next_id}++ }

sub type_max (@types) {
    return $TYPE_BY_RANK[ max map { $RANK{$_} } @types ];
}

sub error ($msg, $node = undef) {
    my $where = $node && $node->{line} ? " at kernel line $node->{line}" : '';
    Carp::croak("PerCUDA: $msg$where");
}

sub walk ($node, $cb) {
    return unless ref $node;
    if (ref $node eq 'ARRAY') {
        walk($_, $cb) for @$node;
        return;
    }
    $cb->($node) if exists $node->{t};
    for my $key (sort keys %$node) {
        next                     if $key eq 'sym';
        walk($node->{$key}, $cb) if ref $node->{$key};
    }
    return;
}

sub is_pure ($e) {
    my $pure = 1;
    walk($e, sub ($n) { $pure = 0 if $n->{t} eq 'assign' || $n->{t} eq 'pre' || $n->{t} eq 'post' });
    return $pure;
}

# Symbols assigned anywhere inside $node (by =, op=, ++ or --).
sub assigned_symbols ($node) {
    my %sym;
    walk(
        $node,
        sub ($n) {
            my $target = $n->{t} eq 'assign' ? $n->{lhs} : ($n->{t} eq 'pre' || $n->{t} eq 'post') ? $n->{e} : undef;
            $sym{ refaddr $target->{sym} } = 1 if $target && $target->{t} eq 'var';
        }
    );
    return \%sym;
}

# ---- argument kinds (array or scalar) are decided by how the kernel uses them ----

sub _analyze_arguments ($self) {
    my (%rank, %scalar, %written, %read);
    my %plain_store;    # elem nodes that are only stored into ($a->[$i] = ...)
    walk(
        $self->{ast},
        sub ($n) {
            my $target = $n->{t} eq 'assign' ? $n->{lhs} : ($n->{t} eq 'pre' || $n->{t} eq 'post') ? $n->{e} : undef;
            if ($target && $target->{t} eq 'elem') {
                $written{ $target->{name} } = 1;
                $plain_store{ refaddr $target } = 1 if $n->{t} eq 'assign' && $n->{op} eq '=';
            }
            if ($n->{t} eq 'elem') {
                my $r = scalar @{ $n->{idx} };
                error("array \$$n->{name} is used with $rank{$n->{name}} and $r subscripts", $n)
                    if $rank{ $n->{name} } && $rank{ $n->{name} } != $r;
                $rank{ $n->{name} } = $r;
                $read{ $n->{name} } = 1 unless $plain_store{ refaddr $n };
            }
            elsif ($n->{t} eq 'var') {
                $scalar{ $n->{name} } = 1;
            }
        }
    );

    my %seen;
    my @args;
    for my $pair (@{ $self->{arguments} }) {
        my ($name, $type) = @$pair;
        error("argument name '$name' is invalid") unless defined $name && $name =~ /\A[A-Za-z_]\w*\z/;
        error("argument '$name' is declared twice") if $seen{$name}++;
        error("argument '$name' has unknown type '" . ($type // '') . "' (use i, f or d)")
            unless defined $type && exists $RANK{$type};
        error("argument name '$name' is reserved") if $self->_is_builtin_name($name);
        push @args,
            {
            name    => $name,
            type    => $type,
            kind    => $rank{$name} ? 'array' : $scalar{$name} ? 'scalar' : 'unused',
            rank    => $rank{$name} // 0,
            read    => $read{$name}    ? 1 : 0,
            written => $written{$name} ? 1 : 0,
            };
    }
    for my $name (sort keys %rank) {
        error("array \$$name is not an argument (kernels cannot declare arrays)") unless $seen{$name};
    }
    $self->{args} = \@args;
    return;
}

sub _is_builtin_name ($self, $name) {
    for my $axis ($self->axes) {
        return 1 if $name eq $axis || $name eq "size_$axis";
    }
    return 0;
}

# ---- scope resolution ----

sub _new_symbol ($self, %args) { return { type => 'i', %args } }

sub _resolve ($self) {
    my %global;
    for my $axis ($self->axes) {
        $global{$axis} = $self->_new_symbol(name => $axis, cname => "percuda_$axis", fixed => 1);
        push @{ $self->{thread_syms} }, $global{$axis};
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
    delete @{$self}{qw(scopes loops)};
    return;
}

sub _lookup ($self, $name, $node) {
    for my $scope (reverse @{ $self->{scopes} }) {
        return $scope->{$name} if $scope->{$name};
    }
    error("variable \$$name is not declared (declare it with my)", $node);
}

sub _declare ($self, $node) {
    my $id  = $self->new_id;
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
    return;
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
        error("'$t' outside of a loop is not supported",   $s) unless $loop;
        error("'$t' inside a bare block is not supported", $s) if $loop eq 'bare';
    }
    elsif ($t ne 'return') {    # return ends the current thread
        error("unsupported statement '$t'", $s);
    }
    return;
}

sub _resolve_loop_body ($self, $body) {
    push @{ $self->{loops} }, 'loop';
    $self->_resolve_block($body);
    pop @{ $self->{loops} };
    return;
}

sub _resolve_expr ($self, $e) {
    my $t = $e->{t};
    if ($t eq 'var') {
        my $sym = $self->_lookup($e->{name}, $e);
        error("array \$$e->{name} must be used with subscripts", $e) if $sym->{array};
        $e->{sym} = $sym;
    }
    elsif ($t eq 'elem') {
        my $sym = $self->_lookup($e->{name}, $e);
        error("\$$e->{name} is not an array argument", $e) unless $sym->{array};
        $e->{sym} = $sym;
        $self->_resolve_expr($_) for @{ $e->{idx} };
    }
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
    elsif ($t ne 'num') {
        error("unsupported expression '$t'");
    }
    return;
}

# ---- type inference: a local is int unless a float/double value is ever assigned to it ----

sub _infer ($self) {
    my @assigns;
    walk(
        $self->{ast},
        sub ($n) {
            if ($n->{t} eq 'my' && $n->{init}) {
                push @assigns, [ $n->{sym}, $n->{init} ];
            }
            elsif ($n->{t} eq 'assign' && $n->{lhs}{t} eq 'var') {
                my $op = $n->{op} =~ s/=\z//r;
                my $rhs =
                      $op eq '' || $op eq '&&' || $op eq '||'
                    ? $n->{rhs}
                    : { t => 'bin', op => $op, l => $n->{lhs}, r => $n->{rhs} };
                push @assigns, [ $n->{lhs}{sym}, $rhs ];
            }
        }
    );
    my $changed = 1;
    while ($changed) {
        $changed = 0;
        for my $a (@assigns) {
            my ($sym, $rhs) = @$a;
            next if $sym->{fixed};
            my $type = type_max($sym->{type}, $self->type_of($rhs));
            if ($type ne $sym->{type}) {
                $sym->{type} = $type;
                $changed = 1;
            }
        }
    }
    return;
}

sub type_of ($self, $e) {
    my $t = $e->{t};
    return $e->{float} ? $self->{real} : 'i'                          if $t eq 'num';
    return $e->{sym}{type}                                            if $t eq 'var' || $t eq 'elem';
    return $self->type_of($e->{lhs})                                  if $t eq 'assign';
    return $self->type_of($e->{e})                                    if $t eq 'pre' || $t eq 'post';
    return type_max($self->type_of($e->{a}), $self->type_of($e->{b})) if $t eq 'cond';
    if ($t eq 'un') {
        return $e->{op} eq '-' ? $self->type_of($e->{e}) : 'i';
    }
    if ($t eq 'call') {
        my @args = map { $self->type_of($_) } @{ $e->{args} };
        return $args[0]      if $e->{name} eq 'abs';
        return 'i'           if $e->{name} eq 'int';
        return $self->{real} if $e->{name} eq 'random';
        return type_max($self->{real}, @args);    # sqrt, exp, log, sin, cos, floor, ceil, atan2
    }
    if ($t eq 'bin') {
        my ($l, $r) = ($self->type_of($e->{l}), $self->type_of($e->{r}));
        my $op = $e->{op};
        return type_max($l, $r) if $op eq '+' || $op eq '-' || $op eq '*';
        return type_max($self->{real}, $l, $r) if $op eq '/';
        if ($op eq '**') {
            return $l if nonneg_int_literal($e->{r});
            return type_max($self->{real}, $l) if $r eq 'i';    # repeated multiplication
            return type_max($self->{real}, $l, $r);    # pow
        }
        return type_max($l, $r) if ($op eq '||' || $op eq '&&') && is_pure($e);
        return 'i';
    }
    error("cannot infer type of '$t'");
}

sub nonneg_int_literal ($e) { return $e->{t} eq 'num' && !$e->{float} && $e->{value} >= 0 }

# ---- which elements the kernel writes (for skipping uploads and splitting work) ----

sub _analyze_writes ($self) {
    my @thread = @{ $self->{thread_syms} };
    my $moved  = assigned_symbols($self->{ast});
    my $fixed  = !grep { $moved->{ refaddr $_ } } @thread;    # $x, $y, $z are never assigned

    my $is_thread_var = sub ($e, $axis) { $e->{t} eq 'var' && $e->{sym} == $thread[$axis] };

    my $own_rows = $fixed;
    walk(
        $self->{ast},
        sub ($n) {
            my $target = $n->{t} eq 'assign' ? $n->{lhs} : ($n->{t} eq 'pre' || $n->{t} eq 'post') ? $n->{e} : undef;
            $own_rows = 0 if $target && $target->{t} eq 'elem' && !$is_thread_var->($target->{idx}[0], 0);
        }
    );
    $self->{writes_own_rows} = $own_rows ? 1 : 0;

    # Top-level stores reached by every thread: no return before them.
    my %covered;
    for my $stmt (@{ $self->{ast}{body} }) {
        my $returns = 0;
        walk($stmt, sub ($n) { $returns = 1 if $n->{t} eq 'return' });
        last if $returns;
        next unless $fixed && $stmt->{t} eq 'expr' && $stmt->{e}{t} eq 'assign' && $stmt->{e}{op} eq '=';
        my $lhs = $stmt->{e}{lhs};
        next unless $lhs->{t} eq 'elem' && @{ $lhs->{idx} } == $self->{dim};
        next if grep { !$is_thread_var->($lhs->{idx}[$_], $_) } 0 .. $self->{dim} - 1;
        $covered{ $lhs->{name} } = 1;
    }
    $_->{covered} = $covered{ $_->{name} } && !$_->{read} ? 1 : 0 for @{ $self->{args} };
    return;
}

# ---- kernel ABI shared by the C generator and the GPU engine ----
#
# Arrays are device pointers; scalars, the grid size and the inner dimensions
# of each array are passed by value:
#
#   percuda_kernel(<arrays and scalars in declaration order>,
#                  size_x [, size_y [, size_z]], <inner dims of each array>)

#
# With $check (check => 1), the outermost dimension of each array follows
# too, and an int *percuda_err buffer comes last.

sub kernel_params ($self, $check = 0) {
    my @params;
    for my $i (0 .. $#{ $self->{args} }) {
        my $arg = $self->{args}[$i];
        next if $arg->{kind} eq 'unused';
        push @params, { kind => $arg->{kind}, arg => $i, type => $arg->{type}, name => $arg->{name} };
    }
    for my $d (0 .. $self->{dim} - 1) {
        push @params, { kind => 'size', axis => $d, type => 'i', name => "size_$AXES[$d]" };
    }
    for my $i (0 .. $#{ $self->{args} }) {
        my $arg = $self->{args}[$i];
        next unless $arg->{kind} eq 'array';
        push @params, { kind => 'dim', arg => $i, axis => $_, type => 'i', name => "d_$arg->{name}_$_" }
            for ($check ? 0 : 1) .. $arg->{rank} - 1;
    }
    push @params, { kind => 'error', name => 'percuda_err' } if $check;
    return \@params;
}

1;
