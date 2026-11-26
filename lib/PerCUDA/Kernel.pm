package PerCUDA::Kernel;
use v5.40;
use B             ();
use Carp          ();
use Symbol        qw(qualify_to_ref);
use List::Util    ();
use File::Spec    ();
use PerCUDA::Data qw(type_code);
use PerCUDA::Analyzer;
use PerCUDA::Parser ();

our @CARP_NOT = qw(PerCUDA PerCUDA::Engine attributes);

# Kernels written as ordinary subs with the :GPU attribute
# (use PerCUDA). The attribute handler reads the body of the sub
# from its source file and replaces the sub with one that runs it as a
# kernel; the runner is built on the first call.

# thread_index() and grid_size(), imported into every package that uses
# PerCUDA so that kernel subs compile under strict. Only kernels give them a
# value (a kernel sub itself never runs as Perl).
my %INDEX_FUNCTION = map {
    my $func = $_;
    $func => sub (@) { Carp::croak("PerCUDA: $func() can only be called in a kernel") }
} sort keys %PerCUDA::Parser::INDEX;

my %OPTION =
    map { $_ => 1 } qw(argument size returns constant precision mode engine check perl_jobs cache_dir device);
my %KERNEL;    # refaddr of a :GPU sub => sub returning its runner
my %SOURCE;    # refaddr of a sub => [ parameter names, body, package ]
my %ATTRIBUTE;    # refaddr of a :GPU sub => arguments of PerCUDA->new from its attribute
my %CONFIG;    # package => PerCUDA::Context from PerCUDA::configure
my @KEEP;    # the :GPU subs, alive for the life of the process (the hashes above use their addresses)

sub import_into ($class, $package) {

    # A sub of the package's own by that name stays (kernels never call it).
    for my $func (sort keys %INDEX_FUNCTION) {
        my $glob = qualify_to_ref($func, $package);
        *$glob = $INDEX_FUNCTION{$func} unless *$glob{CODE};
    }
    my $next = $package->can('MODIFY_CODE_ATTRIBUTES');    # another attribute handler, if any
    no warnings 'redefine';
    *{ qualify_to_ref("MODIFY_CODE_ATTRIBUTES", $package) } = sub ($pkg, $cv, @attrs) {
        my @rest;
        for my $attr (@attrs) {
            if ($attr =~ /\AGPU(?:\((.*)\))?\z/s) { _install($pkg, $cv, $1 // '') }
            else                                  { push @rest, $attr }
        }
        return @rest && $next ? $next->($pkg, $cv, @rest) : @rest;
    };
    return;
}

# :GPU(float, float, int, { precision => 'double', size => [ 'n', 3 ] }):
# the types of the parameters, in order (all of them, or none for the
# default types), then optionally a hash of options of PerCUDA->new, written
# as in Perl (evaluated when the sub is compiled, so only literals; barewords
# are strings; constants of $package can be used).
sub parse_attribute ($text, $package = "main") {
    my @types;
    while ($text =~ s/\A\s*(\w+)\s*(?:,|\z)//) {
        my $type = type_code($1)
            // Carp::croak("PerCUDA: :GPU expects types (int, float or double) and a hash of options, not '$1'");
        push @types, $type;
    }
    return (\@types, {}) unless $text =~ /\S/;
    Carp::croak("PerCUDA: :GPU options must be one hash after the types, e.g. :GPU(float, { precision => 'double' })")
        unless $text =~ /\A\s*\{/;
    my $options = _evaluate($text, $package);
    Carp::croak("PerCUDA: :GPU options must be one hash after the types, e.g. :GPU(float, { precision => 'double' })")
        unless ref $options eq 'HASH';
    for my $key (sort keys %$options) {
        Carp::croak("PerCUDA: unknown :GPU option '$key'") unless $OPTION{$key};
    }
    return (\@types, $options);
}

# The hash of options of :GPU(...), evaluated in the sub's package, so that
# its constants (use constant) can be used; barewords are strings, and a
# variable is an error (it would not have a value yet).
sub _evaluate ($text, $package) {
    my $code = "package $package; use strict; no strict 'subs'; use warnings; no warnings 'reserved'; ($text)";
    my @list = eval $code;    ## no critic (ProhibitStringyEval)
    Carp::croak("PerCUDA: cannot read the :GPU options ($text): $@") if $@;
    return @list == 1 ? $list[0] : undef;
}

# Finds "sub NAME" (or, with $name undef, an anonymous "sub") at or before
# $line in $text and returns (parameter names, body without its braces,
# starting at its line).
sub extract_sub ($text, $name, $line) {
    my $label = defined $name ? "sub $name"             : 'the anonymous sub';
    my $decl  = defined $name ? qr/\bsub\s+\Q$name\E\b/ : qr/\bsub\b(?=\s*[(:{])/;
    my @lines = split /^/, $text;
    my $from;
    for (my $i = $line - 1; $i >= 0; $i--) {
        next unless $i <= $#lines;
        my $code = $lines[$i] =~ s{(?<![\$\@])#.*}{}sr;    # not in a comment ($#array is not one)
        my $count = () = $code =~ /$decl/g;
        next unless $count;

        # The source tells subs apart by line only: two anonymous subs on one
        # line would be indistinguishable.
        Carp::croak("PerCUDA: line " . ($i + 1) . " has $count anonymous subs; put each kernel sub on its own line")
            if $count > 1 && !defined $name;
        $from = $i;
        last;
    }
    Carp::croak("PerCUDA: cannot find $label in its source file") unless defined $from;
    my $src = join '', @lines[ $from .. $#lines ];
    $src =~ /\G.*?$decl\s*/gcs;

    my $balanced = sub ($open, $close) {    # from just after $open to its $close
        my ($depth, $start) = (1, pos $src);

        # $# and @# are not comments
        while ($src =~ /\G(?:[\$\@]\#?|#[^\n]*|[^\Q$open$close\E#\$\@]+|(\Q$open\E)|(\Q$close\E))/gc) {
            $depth++                                            if defined $1;
            $depth--                                            if defined $2;
            return substr($src, $start, pos($src) - $start - 1) if $depth == 0;
        }
        Carp::croak("PerCUDA: unbalanced '$open' in $label");
    };

    my $signature;
    while (1) {
        if ($src =~ /\G:\s*\w+/gc) {    # an attribute; its arguments follow without a space
            $balanced->("(", ")") if $src =~ /\G\(/gc;
            $src =~ /\G\s*/gc;
        }
        elsif ($src =~ /\G\(/gc) { $signature = $balanced->('(', ')'); $src =~ /\G\s*/gc }
        elsif ($src =~ /\G\{/gc) { last }
        else                     { Carp::croak("PerCUDA: cannot parse the declaration of $label") }
    }
    Carp::croak("PerCUDA: kernel $label needs a signature naming its arguments, e.g. sub (\$in, \$out) { ... }")
        unless defined $signature;
    my @names;
    for my $param (grep { length } map { s/\A\s+|\s+\z//gr } split /,/, $signature =~ s/#[^\n]*//gr) {
        Carp::croak("PerCUDA: parameter '$param' of $label must be a plain scalar (no defaults or slurpy)")
            unless $param =~ /\A\$([A-Za-z_]\w*)\z/;
        push @names, $1;
    }

    # Kernel line numbers (in errors) are the lines of the file.
    my $lines_before = $from + (substr($src, 0, pos $src) =~ tr/\n//);
    return (\@names, ("\n" x $lines_before) . $balanced->('{', '}'));
}

# (parameter names, body) of a code reference, read from its source file.
sub source_of ($class, $cv) {
    return @{ $SOURCE{ 0 + $cv } } if $SOURCE{ 0 + $cv };    # a :GPU sub (its original), read when it was compiled
    my $obj  = B::svref_2object($cv);
    my $name = $obj->GV->NAME;
    undef $name if $name eq '__ANON__';
    my $label = defined $name ? "sub $name" : 'the anonymous sub';

    # The first statement of a sub with a signature (its argument check) can
    # carry the line of the enclosing statement, so take the later of the
    # first two statements: it is inside the sub, after "sub".
    my @cops;
    for (my $op = $obj->START; $op && !$op->isa('B::NULL') && @cops < 2; $op = $op->next) {
        push @cops, $op if $op->isa('B::COP');
    }
    Carp::croak("PerCUDA: $label has no body written in Perl") unless @cops;
    my ($file, $line) = ($cops[0]->file, List::Util::max(map { $_->line } @cops));

    # a relative path is relative to the directory perl started in
    if (defined $PerCUDA::START_DIR && !File::Spec->file_name_is_absolute($file)) {
        my $absolute = File::Spec->rel2abs($file, $PerCUDA::START_DIR);
        $file = $absolute if -e $absolute;
    }
    open my $fh, '<', $file or Carp::croak("PerCUDA: cannot read the source of $label ($file): $!");
    my $text = do { local $/; <$fh> };
    close $fh;
    return (extract_sub($text, $name, $line), $cops[0]->stashpv);
}

# Values of the constants (use constant NAME => ...) of $package that the
# kernel ($ast) uses as barewords.
sub package_constants ($class, $package, $ast) {
    my %value;
    for my $name (PerCUDA::Analyzer::constant_names($ast)) {
        my $code = $package->can($name) or next;
        next unless B::svref_2object($code)->CvFLAGS & B::CVf_CONST();    # use constant, not any sub
        my @value = $code->();
        Carp::croak("PerCUDA: constant $name is a list; kernels take one number") unless @value == 1;
        $value{$name} = $value[0];
    }
    return \%value;
}

sub _install ($pkg, $cv, $attribute) {
    my ($types, $options) = parse_attribute($attribute, $pkg);
    my $name = B::svref_2object($cv)->GV->NAME;
    Carp::croak('PerCUDA: :GPU needs a named sub') if $name eq '__ANON__';
    my ($names, $body, $package) = __PACKAGE__->source_of($cv);
    $SOURCE{ 0 + $cv } = [ $names, $body, $package ];
    my %args = (%$options, source => $cv);
    if (@$types) {
        Carp::croak("PerCUDA: sub $name has " . @$names . ' parameters but :GPU gives ' . @$types . ' types')
            unless @$names == @$types;
        Carp::croak("PerCUDA: :GPU gives the types twice (in the list and in 'argument')") if $options->{argument};
        $args{argument} = [ map { $names->[$_] => $types->[$_] } 0 .. $#$names ];
    }

    # Built on the first call, when the package's constants all exist, from
    # the package's configuration (PerCUDA::configure) if any; built again
    # after the configuration changes.
    my ($runner, $built_with, $kernel);
    my $build = sub () {
        my $context = $CONFIG{$pkg};
        if (!$runner || ($built_with // 0) != ($context // 0)) {
            $runner     = $context ? $context->kernel(source => $kernel) : PerCUDA->new(%args);
            $built_with = $context;
        }
        return $runner;
    };
    $kernel                   = sub (@values) { return $build->()->run(@values) };
    $KERNEL{ 0 + $kernel }    = $build;
    $ATTRIBUTE{ 0 + $kernel } = \%args;
    push @KEEP, $kernel;    # never freed, so its address is never reused by another sub
    no warnings 'redefine';
    *{ qualify_to_ref($name, $pkg) } = $kernel;
    return;
}

# PerCUDA::configure: the options (a PerCUDA::Context) of the :GPU subs of
# $package, and of PerCUDA::array called from it.
sub configure ($class, $package, @options) {
    require PerCUDA::Context;
    if (@options == 1 && ref $options[0] && eval { $options[0]->isa('PerCUDA::Context') }) {
        return $CONFIG{$package} = $options[0];
    }
    my %options = @options == 1 && ref $options[0] eq 'HASH' ? %{ $options[0] } : ();
    if (@options == 1 && ref $options[0] && ref $options[0] ne 'HASH') {
        Carp::croak('PerCUDA: configure expects name => value pairs, a hash or a PerCUDA::Context');
    }
    if (!(@options == 1 && ref $options[0])) {
        Carp::croak('PerCUDA: configure expects name => value pairs, a hash or a PerCUDA::Context') if @options % 2;
        %options = @options;
    }
    for my $key (sort keys %options) {
        Carp::croak("PerCUDA: configure cannot set '$key' (it belongs to each kernel)")
            if $key =~ /\A(?:source|argument|size|returns)\z/;
    }
    return $CONFIG{$package} = PerCUDA::Context->new(%options);
}

# The configuration of $package for PerCUDA::array: its own, or a default
# one (which does not configure its :GPU subs).
my %DEFAULT_CONTEXT;

sub package_context ($class, $package) {
    require PerCUDA::Context;
    return $CONFIG{$package} // ($DEFAULT_CONTEXT{$package} //= PerCUDA::Context->new);
}

# The arguments of PerCUDA->new given by the :GPU attribute of $code (with
# source => the original sub), or undef for another sub.
sub attribute_args ($class, $code) { return $ATTRIBUTE{ 0 + $code } }

# The PerCUDA runner behind a :GPU sub (built if needed).

sub runner_of ($class, $code) {
    my $get = $KERNEL{ 0 + $code } // Carp::croak('PerCUDA: not a :GPU sub');
    return $get->();
}

1;
__END__

=encoding utf-8

=head1 NAME

PerCUDA::Kernel - Kernels written as subs with the :GPU attribute

=head1 SYNOPSIS

    use PerCUDA;

    sub add :GPU(float, float, float) ($a, $b, $out) {
        my ($i, $j) = thread_index();
        $out->[$i][$j] = $a->[$i][$j] + $b->[$i][$j];
    }

    my $sum = add($p, $q, []);

=head1 DESCRIPTION

The import of C<use PerCUDA>; see L<PerCUDA/KERNEL SUBS>.

=cut
