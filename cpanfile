requires 'perl', '5.040000';
requires 'Carp';
requires 'Digest::MD5';
requires 'File::Path';
requires 'File::Spec';
requires 'File::Temp';
requires 'List::Util';

# needed only for the gpu engine (https://github.com/papix/CUDA-DriverAPI)
recommends 'CUDA::DriverAPI', '0.02';

on 'test' => sub {
    requires 'Test::More', '0.98';
};
