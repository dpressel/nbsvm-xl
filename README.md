# nbsvm-xl

NBSVM using SGD for fast processing of larger datasets

## More info
Fast out-of-core SGD SVM or LR demonstration code for:

Sida Wang and Christopher Manning. "Simple, good sentiment and topic classification"

This code depends on my sgdtk Java library for its SVM/LR SGD implementations, which also use overlapped File IO and
feature hashing.  Most parameters that would be useful are parameterized here, both from main, and as setters.

Training and test data are essentially TSVs with the first column containing -1 (negative) or 1 (positive) label and
space delimited tokens in the second column.