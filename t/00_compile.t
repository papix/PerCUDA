use v5.40;
use Test::More;

use_ok $_ for qw(
    PerCUDA
    PerCUDA::Parser
    PerCUDA::CodeGen
    PerCUDA::Compiler
    PerCUDA::Data
    PerCUDA::Engine::Perl
    PerCUDA::Engine::GPU
);

done_testing;
