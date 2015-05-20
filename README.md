# nbsvm-xl

NBSVM using SGD for fast processing of larger datasets

## More info
Fast, scalable SGD SVM or LR demonstration code for:

Sida Wang and Christopher Manning. "Simple, good sentiment and topic classification"

I wanted to demonstrate the simplicity of this algorithm with minimal outside dependencies, while demonstrating
how an SGD implementation can be used effectively for this type of task, allowing it to scale to larger datasets.
On my laptop, this command line program can process a million tweets including the lexicon in well under two minutes,
with the first epoch taking around 45s, and additional epochs at around 15s.  This is much faster than it can be
done using liblinear, and due to how it handles IO (based on Vowpal Wabbit), it should scale much better as well.

Supports generative features as in the paper, as well as NB classification interpolation as described.  You
can pick whether to use SVM or LR by using the --loss option.  You can control the NB interpolation using --beta
and the number of N-gram shingles using --ngram.  Smoothing on the generative features is controlled with the --alpha parameter.
A whole host of other parameters are available. -- see the usage for an exhaustive list -- everything is tunable.

This code depends on my sgdtk Java library for its SVM/LR SGD implementations (https://github.com/dpressel/sgdtk), 
which also use overlapped File IO and feature hashing.  A key feature is speed of training, even with large
amounts of training data.  The ring buffer size options control how much processing may be done in-core.

Training and test data are essentially TSVs with the first column containing -1 (negative) or 1 (positive) label and
space delimited tokens in the second column.

To build NBSVM, first pull https://github.com/dpressel/sgdtk and do a maven install.  Then run the maven
build for this project, which will then find it in your local cache.

You may use this code for commercial or other purposes, but please attribute your work to this implementation if you do.
