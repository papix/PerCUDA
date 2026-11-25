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
    PerCUDA::Engine::Perl
    PerCUDA::Engine::GPU
);

for my $module (@modules) {
    ok lives { eval "require $module; 1" or die $@ }, "load $module" or note $@;
}

done_testing;
