[install]
pip3.5 install Theano numpy matplotlib


[run]
env THEANO_FLAGS=device=gpu,floatX=float32 python3.5 test-chem.py {sample_file_name} {weight_file_name}
