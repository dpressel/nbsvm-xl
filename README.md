# nbsvm-xl

NBSVM using SGD for fast processing of large datasets

## More info
Fast, scalable SGD SVM or LR demonstration code for:

[Sida Wang](https://github.com/sidaw) and Christopher Manning. "Simple, good sentiment and topic classification"

I wanted to demonstrate the simplicity of this algorithm with minimal outside dependencies, while demonstrating
how an SGD implementation can be used effectively for this type of task, allowing it to scale to larger datasets.
On my laptop, this command line program can process 10 epochs of 1.6 million tweets including the lexicon in 2 1/2 minutes total,
with the first epoch taking around 45s, and additional epochs at around 8s.  This is much faster than it can be
done using liblinear, and due to how it handles IO and feature vectors, it should scale much better as well.  You can find some more information on many of the techniques used for nbsvm-xl used here:

https://rawgit.com/dpressel/Meetups/master/nlp-meetup-2016-02-25/presentation.html

Supports generative features as in the paper, as well as NB classification interpolation as described.  You
can pick whether to use SVM or LR by using the --loss option.  You can control the NB interpolation using --beta
and the number of N-gram shingles using --ngram.  You can optionally add shingled char ngram support (not in the orignal paper)
using --cgram values > 0.  Smoothing on the generative features is controlled with the --alpha parameter.
A whole host of other parameters are available. -- see the usage for an exhaustive list -- everything is tunable.
Adagrad is currently optional for optimization using --method adagrad

## Unique features

This code implements several features that, as far as I know, are unique among NBSVM implementations.  Here are unique features of nbsvm-xl

  - Scales to huge datasets with stable memory footprint and fast processing times
  - "Long distance" lexicon support
    - Use a different lexicon to create the generative feature values. Works best with a large corpus correlated with truth, but not actual truth, in addition to a small ground truth training set. In this case, the ground truth may be too small to get good MLEs for the generative features.  When this method improves performance over just ground truth, it also typically will produce superior performance to just using the correlated dataset as ground truth.

  - Supports character ngrams
    - Useful for processing datasets like Twitter, where words can be part of hashtags, mentions or otherwise convoluted, or where morphology is important

  - Uses feature hashing 

  - Supports NB smoothing, as in the original paper.  Some other implementations do as well, but the most widely used implementation does not

  - Supports generating an NB parameter that is added to the vector.  This occasionally improves performance

## Details
This code depends on my sgdtk Java library for its SVM/LR SGD implementations (https://github.com/dpressel/sgdtk).  A key feature is speed of training, even with large amounts of training data.  The ring buffer size options control how much processing may be done in-core.

To build nbsvm-xl, first pull https://github.com/dpressel/sgdtk and do a maven install.  Then run the maven
build for this project, which will then find it in your local cache.

Training and test data are essentially TSVs with the first column containing -1 (negative) or 1 (positive) label and
space delimited tokens in the second column.  On linux its fairly easy to get data that was downloaded using the _oh_my_gosh.sh_ script from https://github.com/mesnilgr

Preprocess the files:
```
awk '{ print "-1\t",$0 }' test-neg.txt > test-neg.tsv
awk '{ print "-1\t",$0 }' train-neg.txt > train-neg.tsv
awk '{ print "1\t",$0 }' test-pos.txt > test-pos.tsv
awk '{ print "1\t",$0 }' train-pos.txt > train-pos.tsv

cat train-neg.tsv train-pos.tsv > unshuf.tsv
shuf unshuf.tsv > train-xl.tsv
cat test-neg.tsv test-pos.tsv > unshuf.tsv
shuf unshuf.tsv > test-xl.tsv
rm -f unshuf.tsv

```

Run using nbsvm.py

```
dpressel@dpressel:~/dev/work/nbsvm_run$ python ../nbsvm/nbsvm.py --liblinear liblinear-1.96 --ptrain data/train-pos.txt --ntrain data/train-neg.txt --ptest data/test-pos.txt --ntest data/test-neg.txt --ngram 123 --out NBSVM-TEST-TRIGRAM
counting...
computing r...
processing files...
iter  1 act 1.236e+04 pre 1.070e+04 delta 8.596e+00 f 1.733e+04 |g| 7.848e+03 CG   7
iter  2 act 3.132e+03 pre 2.542e+03 delta 1.033e+01 f 4.970e+03 |g| 2.126e+03 CG   9
iter  3 act 9.326e+02 pre 7.520e+02 delta 1.033e+01 f 1.838e+03 |g| 7.607e+02 CG   9
iter  4 act 2.631e+02 pre 2.130e+02 delta 1.033e+01 f 9.055e+02 |g| 2.825e+02 CG   8
iter  5 act 7.708e+01 pre 6.203e+01 delta 1.033e+01 f 6.424e+02 |g| 1.066e+02 CG   7
iter  6 act 3.909e+01 pre 3.054e+01 delta 1.033e+01 f 5.653e+02 |g| 4.129e+01 CG   9
Accuracy = 91.872% (22968/25000)
```

Now using nbsvm-xl

```
xl.nbsvm.NBSVM --train /home/dpressel/dev/work/nbsvm_run/data/train-xl.tsv --eval /home/dpressel/dev/work/nbsvm_run/data/test-xl.tsv 
--cgrams 0 --ngrams 3 --nbits 26 --loss log --epochs 10 -e0 0.05 --lambda 1e-6 --beta 1

4951537 hash words in lexicon, aggregated in 6.72s.  Starting training
Trained model in 18.81s.  25000 training examples seen
22969 / 25000
Model accuracy 91.88 %
```

Note in this example NB regularization is turned off altogether since the python code doesnt support that.


You may use this code for commercial or other purposes, but please attribute your work to this implementation if you do.

