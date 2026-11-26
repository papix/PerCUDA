package PerCUDATest::Configured;
use v5.40;
use PerCUDA ':kernel' => { engine => 'perl', precision => 'double' };

# :GPU subs configured when :kernel is imported (t/16_gpu_attribute.t).

sub third : GPU ($out) {
    $out->[$x] = 1.0 / 3.0;
}

sub array_here (@args) { return PerCUDA->array(@args) }

1;
