#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools
samples=$base/samples

mkdir -p $samples

num_threads=8

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/cornell \
        --words 200 \
        --checkpoint $models/model_dropout0.2.pt \
        --temperature 0.7 \
        --mps \
        --outf $samples/sample_dropout0.2.txt
)
