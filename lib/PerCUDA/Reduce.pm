package PerCUDA::Reduce;
use v5.40;
use Carp       ();
use List::Util ();

# Reductions of a whole array (sum, min, max, argmin, argmax) in a fixed
# order, so that the perl engine can replay the GPU's floating point sums bit
# for bit:
#
#   pass 1: BLOCKS blocks of THREADS threads; thread t folds the elements
#           t, t + T, t + 2T, ... (T = BLOCKS * THREADS) in order, then each
#           block combines its threads pairwise (s = THREADS/2, ..., 1)
#   pass 2: one block folds the BLOCKS partial results the same way
#
# BLOCKS is ceil(n / THREADS), at most MAX_BLOCKS. Integer sums are 64-bit.
# min and max ignore nan unless every element is nan; ties go to the lower
# index.

our $THREADS    = 256;
our $MAX_BLOCKS = 1024;
our @OPS        = qw(sum min max argmin argmax);

my %CTYPE = (i => 'int',       f => 'float', d => 'double');
my %ACC   = (i => 'long long', f => 'float', d => 'double');

sub blocks ($n) {
    my $blocks = int(($n + $THREADS - 1) / $THREADS);
    return $blocks < 1 ? 1 : $blocks > $MAX_BLOCKS ? $MAX_BLOCKS : $blocks;
}

sub kernel_name ($op, $type, $pass) { return "percuda_reduce_${op}_${type}_$pass" }

# CUDA C++ for both passes of one operation and type.
sub cuda_source ($op, $type) {
    my $in  = $CTYPE{$type};
    my $acc = $op eq 'sum' ? $ACC{$type} : $CTYPE{$type};
    my $combine =
        $op eq 'sum'
        ? 'if (ai < 0) { a = v; ai = 0; } else if (vi >= 0) { a = a + v; }'
        : do {
        my $better = $op =~ /min/ ? '(v < a || (a != a && v == v))' : '(v > a || (a != a && v == v))';
        "if (vi >= 0 && (ai < 0 || $better || (v == a && vi < ai))) { a = v; ai = vi; }";
        };
    my $src = '';
    for my $pass (1, 2) {
        my $vin  = $pass == 1 ? $in : $acc;
        my $name = kernel_name($op, $type, $pass);
        my $idx  = $pass == 1 ? 'k' : 'in_idx[k]';
        $src .= <<"CUDA";
extern "C" __global__ void $name(const $vin *__restrict__ in, const int *__restrict__ in_idx, int n,
                                $acc *__restrict__ out, int *__restrict__ out_idx) {
    __shared__ $acc s[$THREADS];
    __shared__ int si[$THREADS];
    int tid = threadIdx.x;
    int stride = gridDim.x * $THREADS;
    $acc a = 0;
    int ai = -1;
    for (int k = blockIdx.x * $THREADS + tid; k < n; k += stride) {
        $acc v = ($acc)in[k];
        int vi = $idx;
        $combine
    }
    s[tid] = a;
    si[tid] = ai;
    for (int h = $THREADS / 2; h > 0; h >>= 1) {
        __syncthreads();
        if (tid < h) {
            a = s[tid];
            ai = si[tid];
            $acc v = s[tid + h];
            int vi = si[tid + h];
            $combine
            s[tid] = a;
            si[tid] = ai;
        }
    }
    if (tid == 0) {
        out[blockIdx.x] = s[0];
        out_idx[blockIdx.x] = si[0];
    }
}
CUDA
    }
    return $src;
}

# The same computation in Perl. $round rounds single precision results.
# Returns (value, index); index is -1 for an empty array.
sub replay ($values, $op, $type, $round) {
    my $f32 = $round && $type eq 'f';
    my $add = $f32 ? sub ($x, $y) { unpack 'f', pack 'f', $x + $y } : sub ($x, $y) { $x + $y };
    my $better =
          $op eq 'sum' ? undef
        : $op =~ /min/ ? sub ($v, $acc) { $v < $acc || ($acc != $acc && $v == $v) }
        :                sub ($v, $acc) { $v > $acc || ($acc != $acc && $v == $v) };
    my $combine = sub ($acc, $acc_i, $v, $vi) {
        if ($op eq 'sum') {
            return ($v,                                 0) if $acc_i < 0;
            return ($vi >= 0 ? $add->($acc, $v) : $acc, $acc_i);
        }
        return ($v,   $vi) if $vi >= 0 && ($acc_i < 0 || $better->($v, $acc) || ($v == $acc && $vi < $acc_i));
        return ($acc, $acc_i);
    };
    my $fold = sub ($vals, $idxs, $blocks) {    # one launch: returns the per-block results
        my $n      = @$vals;
        my $stride = $blocks * $THREADS;
        my (@out, @out_idx);
        for my $block (0 .. $blocks - 1) {
            my (@s, @si);
            for my $tid (0 .. $THREADS - 1) {
                my ($acc, $acc_i) = (0, -1);
                for (my $k = $block * $THREADS + $tid; $k < $n; $k += $stride) {
                    ($acc, $acc_i) = $combine->($acc, $acc_i, $vals->[$k], $idxs ? $idxs->[$k] : $k);
                }
                ($s[$tid], $si[$tid]) = ($acc, $acc_i);
            }
            for (my $h = $THREADS / 2; $h > 0; $h >>= 1) {
                ($s[$_], $si[$_]) = $combine->($s[$_], $si[$_], $s[ $_ + $h ], $si[ $_ + $h ]) for 0 .. $h - 1;
            }
            push @out,     $s[0];
            push @out_idx, $si[0];
        }
        return (\@out, \@out_idx);
    };
    my @vals = $type eq 'f' && $f32 ? map { unpack 'f', pack 'f', $_ // 0 } @$values : map { $_ // 0 } @$values;
    my ($partial, $partial_idx) = $fold->(\@vals,   undef,        blocks(scalar @vals));
    my ($result,  $result_idx)  = $fold->($partial, $partial_idx, 1);
    return ($result->[0], $result_idx->[0]);
}

# Fast version for mode => 'fast': Perl's own order.
sub quick ($values, $op) {
    return ($op eq 'sum' ? 0 : undef,   -1) unless @$values;
    return (List::Util::sum0(@$values), 0) if $op eq 'sum';
    my $best = 0;
    for my $k (1 .. $#$values) {
        my ($v, $b) = ($values->[$k], $values->[$best]);
        $best = $k if $op =~ /min/ ? ($v < $b || ($b != $b && $v == $v)) : ($v > $b || ($b != $b && $v == $v));
    }
    return ($values->[$best], $best);
}

1;
