#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools

mkdir -p $models

num_threads=8
dropouts=(0.0 0.2 0.3 0.4 0.5)



for dropout in "${dropouts[@]}"; do
    echo "Training with dropout = $dropout"
    SECONDS=0
	(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/cornell \
        --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout $dropout --tied \
        --mps \
        --save $models/model_dropout${dropout}.pt \
        --ppl-log $models/dropout${dropout}.tsv \
	> $models/log_dropout${dropout}.txt
)

echo "✅ Done: dropout=$dropout, time taken: $SECONDS seconds"
    echo "------------------------------------------------------"
done
