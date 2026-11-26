package PerCUDATest::KernelSubs;
use v5.40;
use PerCUDA ':kernel';

# :GPU subs in a module loaded through @INC (t/16_gpu_attribute.t).

my $unrelated = 'sub scale is mentioned here, before the real one';

sub scale : GPU(float, float, float, { retval => [] }) ($in, $out, $s) {
    $out->[$x] = $in->[$x] * $s;
}

sub twice : GPU ($out, $in) {
    $out->[$x] = $in->[$x] * 2;
}

1;
