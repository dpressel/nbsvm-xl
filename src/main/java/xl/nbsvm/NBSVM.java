package xl.nbsvm;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.sgdtk.*;
import org.sgdtk.exec.OverlappedTrainingRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Perform NBSVM (supporting shingles up to 4, and both SVM and LR implementations using SGD and feature hashing).
 * Test files should be binary, where the label will be -1 or 1, followed by a space, and then a sentence
 * Sentences are tokenized by splitting on whitespace.
 *
 * -1 hate this stupid car
 * 1 nbsvm is great !
 *
 * <p>
 * For more information, the original paper:
 * Sida Wang and Christopher Manning. "Simple, good sentiment and topic classification"
 * <p/>
 *
 * @author dpressel
 */
public class NBSVM
{

    private static final Logger log = LoggerFactory.getLogger(NBSVM.class);

    public NBSVM()
    {
    }


    enum LossType {LOG, LR, HINGE }

    public static Loss lossFor(String loss)
    {
        LossType lossType = LossType.valueOf(loss.toUpperCase());
        log.info("Using " + lossType.toString() + " loss function");
        if (lossType == LossType.LR || lossType == LossType.LOG)
        {
            return new LogLoss();
        }
        return new HingeLoss();
    }

    public static class Params
    {

        // This code supports Long Distance lexicons.  As far as I know, there is no other implementation of NBSVM
        // that allows this.  An example where this proves useful would be cases where you have a large corpus highly
        // correlated with the true label, but is not fully supervised, and your normal traininig set, which is much
        // smaller than you wanted.  In that case, you can use this option to pass the big corpus in, and the small one
        // to learn the training activation.  This can significantly increase performance, as the wordstats on the small
        // corpus will be sparse.  Interestingly, this approach often works better than just using the bigger corpus
        // as training data -- also its much faster and more efficient
        @Parameter(description = "Lexicon corpus", names = {"--lex", "-lex"})
        public String lex;

        @Parameter(description = "Training file", names = {"--train", "-t"}, required = true)
        public String train;

        @Parameter(description = "Testing file", names = {"--eval", "-e"}, required = true)
        public String eval;

        @Parameter(description = "Model to write out", names = {"--model", "-s"})
        public String model;


        // Even though this defaults to HINGE, Mesnil et. al. showed that log loss often outperforms
        // I have found this to also be the case on almost every dataset I've ever run
        @Parameter(description = "Loss function", names = {"--loss", "-l"})
        public String loss = LossType.HINGE.toString();

        @Parameter(description = "lambda", names = {"--lambda", "-lambda"})
        public Double lambda = 1e-5;

        @Parameter(description = "eta0, if not set, try and preprocess to find", names = {"--eta0", "-e0"})
        public Double eta0 = 0.5;

        @Parameter(description = "Number of epochs", names = {"--epochs", "-epochs"})
        public Integer epochs = 10;

        // This parameter essentially tells this code how many training instances it may keep in memory at one time
        // Normally, this code is fast enough that these buffer limits wont get hit
        @Parameter(description = "Ring buffer size", names = {"--bufsz"})
        public Integer bufferSz = 16384;

        // This version of NBSVM is the only version Im aware of that uses feature hashing.  This parameter says
        // how many bits to allocate for hashing.
        @Parameter(description = "Number of bits for feature hashing", names = {"--nbits"})
        public Integer nbits = 24;

        @Parameter(description = "N-grams", names = {"--ngrams"})
        public Integer ngrams = 3;

        // Adagrad sometimes performs better than SGD, though this really seems to depend on the set
        @Parameter(description = "Learning method (sgd|adagrad)", names = {"--method"})
        public String method = "sgd";

        // This feature is unique to this version of NBSVM, and is extremely useful for processing datasets like Twitter
        // where the words occur in unusual ways.  With a big enough corpus, the usefulness seems to fall off
        @Parameter(description = "Char N-grams", names = {"--cgrams"})
        public Integer charNgrams = 0;

        @Parameter(description = "SVM to Naive bayes interpolation factor (1 means all SVM, 0 all NB)", names = {"--beta"})
        public Double beta = 0.95;

        @Parameter(description = "Control generative feature smoothing", names = {"--alpha"})
        public Double alpha = 1.0;

        @Parameter(description = "Cache dir", names = {"--cache-dir"})
        public String cacheDir = "cache";

        @Parameter(description = "Minimum sentence length", names = {"--min-sent"})
        public Integer minSentenceLength = 0;

        @Parameter(description = "Lower case tokens", names = "--lower")
        public Boolean lowerCase = false;

        @Parameter(description = "Clean string", names = "--clean")
        public Boolean clean = false;

        // This creates a global feature containing the Naive Bayes calculation (PMI score)
        // This occasionally, helps but you could also just lower alpha if you want NB smoothing
        @Parameter(description = "Use NB parameter", names = "--nbfeat")
        public Boolean useNBFeature = false;

        // This basically tells the code *not to do NBSVM*  You almost never want this
        // it just exists so you can easily compare NBSVM performance against a plain SVM
        // Its a convenience thing
        @Parameter(description = "Use NB parameter", names = "--nonbsvm")
        public Boolean noNBSVM = false;
    }

    public static void main(String[] args)
    {
        try
        {
            Params params = new Params();
            JCommander jc = new JCommander(params, args);
            jc.parse();
            boolean isAdagrad = "adagrad".equals(params.method);
            Classifier nbsvm = new Classifier(params.nbits, params.ngrams, params.charNgrams, params.beta, params.lowerCase, params.clean, params.useNBFeature);

            if (params.ngrams == 0 && params.charNgrams == 0)
            {
                throw new Exception("Must supply at least one word or char ngram");
            }


            Iterator<Instance> lexIterator = new FileIterator(new File(params.lex == null ? params.train: params.lex), params.minSentenceLength);
            Iterator<Instance> testIterator = new FileIterator(new File(params.eval), params.minSentenceLength);
            Iterator<Instance> trainIterator = new FileIterator(new File(params.train), params.minSentenceLength);

            nbsvm.fit(lossFor(params.loss), params.epochs, params.bufferSz, params.lambda, params.eta0,
                    lexIterator, trainIterator, params.alpha, params.noNBSVM, isAdagrad, new File(params.cacheDir));

            if (params.model != null)
            {
                nbsvm.save(params.model);
            }

            double acc = nbsvm.eval(testIterator);

            System.out.println(String.format("Model accuracy %.02f %%", acc * 100));
            System.out.println("Total hashing collisions " + nbsvm.getCollisions());


        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
    }
}
