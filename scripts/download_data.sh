#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# link default training data for easier access

mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# download a different interesting data set!

mkdir -p $data/cornell

mkdir -p $data/cornell/raw

python3 - <<END
from convokit import Corpus, download
corpus = Corpus(filename=download("movie-corpus"))
with open("$data/cornell/raw/utterances.txt", "w", encoding="utf-8") as f:
    for i, utt in enumerate(corpus.iter_utterances()):
        if i >= 10000:
            break
        f.write(utt.text.strip() + "\\n")
END

# preprocess slightly

cat $data/cornell/raw/utterances.txt | python $base/scripts/preprocess_raw.py > $data/cornell/raw/utterances.cleaned.txt

# tokenize, fix vocabulary upper bound

cat $data/cornell/raw/utterances.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    $data/cornell/raw/utterances.preprocessed.txt

# split into train, valid and test

head -n 1000 $data/cornell/raw/utterances.preprocessed.txt > $data/cornell/valid.txt
head -n 2000 $data/cornell/raw/utterances.preprocessed.txt | tail -n 1000 > $data/cornell/test.txt
tail -n 8200 $data/cornell/raw/utterances.preprocessed.txt  > $data/cornell/train.txt
