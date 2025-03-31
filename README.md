Group member: Mingrou Wei 23-739-436 / Yiting Zhao 23-744-170

# MT Exercise 2: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place:

    git clone https://github.com/marpng/mt-exercise-02
    cd mt-exercise-02

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh
    
# I changed the data grasp method for my interested corpus 'Cornell Movie Dialogs Corpus'.
python3 - <<END
from convokit import Corpus, download
corpus = Corpus(filename=download("movie-corpus"))
with open("$data/cornell/raw/utterances.txt", "w", encoding="utf-8") as f:
    for i, utt in enumerate(corpus.iter_utterances()):
        if i >= 10000:
            break
        f.write(utt.text.strip() + "\\n")
END

# I changed the file name into 'cornell' and output name from 'tales' into 'utterances' 

# I found there are some HTML entity coding in 'utterance.preprocessed.txt' like `&apos;` `&quot;` `&amp;`
so I add 'output_string = html.unescape(" ".join(output_tokens))' and 'import HTML' in 'preprocess.py' to avoid HTML entity encodings appear

# Adjusted the data split: 8200 training sentences, 1000 validation, and 1000 test.

Train a model:

    ./scripts/train.sh
    
# I'm using MacOS GPU training the language model. I changed core number num_threads=4 into 8 and enabled Mac MPS backend (--mps) .
# These settings vastly improve my training time within 78s and decrease the ppl to 64.71.
# In the second task, I modify train.sh into train_dropout_experiment.sh to train models with 5 different dropout numbers(0.0 0,2 0.3 0.4 and 0.5)
# After running this file, 5 logfiles, models and csv file containing valid.perplexity of each epoch number with different dropouts are generated and saved in 'models' file


The training process can be interrupted at any time, and the best checkpoint will always be saved.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh
# In generated.sh file, I changed word length into 200 and temperature 0.7, which may improve the coherence and fluency of the generated text.
# In the second task, I modify the generate.sh into generate_0.4_dropout.sh and generate_0.2_dropout.sh to see the text generated with different dropout models
# And the results are in 'samples'file


    /tools/pytorch-examples/word language model/main.py 
# I modify the python file for it can accepts and additional flag to save the perplexities as a log-file

    ./scripts/plot_dropout_results.py
# This Python script combines the 5 `.tsv` files (each containing validation perplexities from a training run with a different dropout value) into a single CSV table, 
and generates a line plot that visualizes the performance comparison over epochs.


Command line
# 1. Train 5 models with different dropout values
    ./scripts/train_dropout_experiments.sh
# 2. Create perplexity table + plot
     python3 scripts/plot_dropout_results.py
# 3. Generate text from best model (lowest test perplexity)
    ./scripts/generate_0.4_dropout.sh
# 4. Generate text from worst model (highest test perplexity)
    ./scripts/generate_0.2_dropout.sh



