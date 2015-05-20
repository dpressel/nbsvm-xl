# nbsvm-xl

NBSVM using SGD for fast processing of larger datasets

## More info
Fast, scalable SGD SVM or LR demonstration code for:

Sida Wang and Christopher Manning. "Simple, good sentiment and topic classification"

Supports generative features as in the paper, as well as NB classification interpolation as described.  You
can pick whether to use SVM or LR by using the --loss option.  You can control the NB interpolation using --beta
and the number of N-gram shingles using --ngram.  Smoothing on the generative features is controlled with the --alpha parameter.
A whole host of other parameters are available. -- see the usage for an exhaustive list -- everything is tunable.

This code depends on my sgdtk Java library for its SVM/LR SGD implementations, which also use overlapped File IO and
feature hashing.  A key feature is speed of training, even with large amounts of training data.  The ring buffer size
options control how much processing may be done in-core.

Training and test data are essentially TSVs with the first column containing -1 (negative) or 1 (positive) label and
space delimited tokens in the second column.