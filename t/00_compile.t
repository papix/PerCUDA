use v5.40;
use Test2::V0;

my @modules = qw(
    PerCUDA
    PerCUDA::Parser
    PerCUDA::Analyzer
    PerCUDA::CodeGen::CUDA
    PerCUDA::CodeGen::Perl
    PerCUDA::Engine
    PerCUDA::Compiler
    PerCUDA::Data
    PerCUDA::Array
    PerCUDA::Engine::Perl
    PerCUDA::Engine::GPU
    PerCUDA::Context
    PerCUDA::Kernel
);

my @warnings;
local $SIG{__WARN__} = sub { push @warnings, @_ };
for my $module (@modules) {
    ok lives { eval "require $module; 1" or die $@ }, "load $module" or note $@;
}
is \@warnings, [], 'no warnings while loading';

done_testing;
