package xl.nbsvm;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
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
    private SGDLearner learner;
    private OverlappedTrainingRunner trainingRunner;
    private int collisions = 0;
    private long hashWordsProcessed = 0;
    private long numPositiveTrainingExamples = 0;
    private long numNegativeTrainingExamples = 0;
    private File cacheDir;
    private Loss loss;
    private int epochs;
    private int bufferSz;
    private double lambda;
    private Map<Integer, Double> lexicon;
    private int ngrams;
    private int nbits;
    private String learningMethod;
    private double eta0;
    private double beta;
    private int charNGramWidth = 0;

    private HashFeatureEncoder hashFeatureEncoder;

    private static final Logger log = LoggerFactory.getLogger(NBSVM.class);


    public int getCollisions()
    {
        return collisions;
    }

    public void setCacheDir(File cacheDir)
    {
        this.cacheDir = cacheDir;
    }

    public void setLoss(Loss loss)
    {
        this.loss = loss;
    }

    public void setEpochs(int epochs)
    {
        this.epochs = epochs;
    }

    public void setBufferSz(int bufferSz)
    {
        this.bufferSz = bufferSz;
    }

    public void setLearningMethod(String method)
    {
        this.learningMethod = method;
    }

    public void setLambda(double lambda)
    {
        this.lambda = lambda;
    }

    public void setNgrams(int ngrams)
    {
        this.ngrams = ngrams;
    }


    public void setNbits(int nbits)
    {
        this.nbits = nbits;
        this.hashFeatureEncoder = new HashFeatureEncoder(nbits);
    }

    public void setEta0(double eta0)
    {
        this.eta0 = eta0;
    }

    /**
     * Get the number of unique words processed
     */
    public long getHashWordsProcessed()
    {
        return hashWordsProcessed;
    }

    /**
     * Get the total number of training examples seen
     */
    public long getNumTrainingExamples()
    {
        return numPositiveTrainingExamples + numNegativeTrainingExamples;
    }

    /**
     * Set how much weight the linear classifier gets in the decision
     * @param beta When set to 1, all weight is on the linear classifer, 0 means all on NB
     */
    public void setBeta(double beta)
    {
        this.beta = beta;
    }

    public int getCharNGramWidth()
    {
        return charNGramWidth;
    }

    public void setCharNGramWidth(int charNGramWidth)
    {
        this.charNGramWidth = charNGramWidth;
    }

    /**
     * Simple view of a single training instance containing only
     * a label and tokenized text
     */
    public static class Instance
    {
        int label;
        List<String> text;
    }

    /**
     * Train the model on the data in the corpus.  This method assumes that the
     * lexicon has already been built, as it is required for feature extraction
     * @param corpus An iterator over the corpus data
     */
    public Model train(Iterator<Instance> corpus) throws Exception
    {

        while (corpus.hasNext())
        {

            Instance instance = corpus.next();
            
            // Used for the prior
            if (instance.label > 0)
            {
                numPositiveTrainingExamples++;
            }
            else
            {
                numNegativeTrainingExamples++;
            }

            if (learner == null)
            {
                onNewTrainingSet();
            }

            FeatureVector fv = transform(instance);

            if (fv == null)
            {
                continue;
            }
            // Add to training set asynchronously
            trainingRunner.add(fv);
        }

        // Wait until training is done
        return trainingRunner.finish();
    }

    // Look up the generative feature
    private void toFeature(Set<Integer> set, List<Offset> offsets, String... str)
    {
        String joined = CollectionsManip.join(str, "_*_"); // This fun delimiter borrowed from Mesnil's implementation
        int idx = hashFeatureEncoder.indexOf(joined);

        if (!set.contains(idx))
        {
            set.add(idx);
            Double v = lexicon.get(idx);
            if (v == null)
            {
                return;
            }
            offsets.add(new Offset(idx, v));
        }
        else
        {
            collisions++;
        }
    }

    // This really should be working over the whole post right??
    private void extractCharGrams(Set<Integer> set, List<Offset> offsets, List<String> tokens)
    {
        if (charNGramWidth == 0)
        {
            return;
        }

        String t = String.join("+", tokens);

        int sz = t.length();

        for (int i = 2; i < charNGramWidth; ++i)
        {
            for (int j = 0; j < sz - i + 1; ++j)
            {
                String sub = t.substring(j , j + i);
                toFeature(set, offsets, sub);
            }
        }
    }

    // This really should be working over the whole post right??
    private void lexCharGrams(List<String> text, Map<Integer, Long> ftable, Long[] numTokens, int labelIdx)
    {
        if (charNGramWidth == 0)
        {
            return;
        }

        String t = String.join("+", text);

        int sz = t.length();

        for (int i = 2; i < charNGramWidth; ++i)
        {
            for (int j = 0; j < sz - i + 1; ++j)
            {
                String sub = t.substring(j , j + i);
                increment(ftable, sub);
                numTokens[labelIdx]++;
            }
        }
    }

    private void lexWordGrams(List<String> text, Map<Integer, Long> ftable, Long[] numTokens, int labelIdx)
    {
        if (ngrams == 0)
        {
            return;
        }
        // Up to 4-grams, nobody needs more than that
        String t = null;
        String l = null;
        String ll = null;
        String lll;

        for (int i = 0, sz = text.size(); i < sz; ++i)
        {
            // Circular
            lll = ll;
            ll = l;
            l = t;
            t = text.get(i);
            // unigram
            increment(ftable, t);
            numTokens[labelIdx]++;
            // bigram?
            if (ngrams > 1 && l != null)
            {
                // trigram?
                increment(ftable, l, t);
                numTokens[labelIdx]++;
                if (ngrams > 2 && ll != null)
                {
                    increment(ftable, ll, l, t);
                    numTokens[labelIdx]++;
                }
                // 4-gram?
                if (ngrams > 3 && lll != null)
                {
                    increment(ftable, lll, ll, l, t);
                    numTokens[labelIdx]++;
                }
            }
        }
    }
    private void extractWordGrams(Set<Integer> set, List<Offset> offsets, List<String> text)
    {
        if (ngrams == 0)
        {
            return;
        }
        // Up to 4-grams, nobody needs more than that
        String t = null;
        String l = null;
        String ll = null;
        String lll;

        for (int i = 0, sz = text.size(); i < sz; ++i)
        {
            // Circular
            lll = ll;
            ll = l;
            l = t;
            t = text.get(i);

            // unigram
            toFeature(set, offsets, t);

            // bigram?
            if (ngrams > 1 && l != null)
            {
                // trigram?
                toFeature(set, offsets, l, t);
                if (ngrams > 2 && ll != null)
                {
                    toFeature(set, offsets, ll, l, t);
                }
                // 4-gram?
                if (ngrams > 3 && lll != null)
                {
                    toFeature(set, offsets, lll, ll, l, t);
                }
            }
        }
    }
    // instance should be tabulated already so just hash the keys and set the values
    private FeatureVector transform(Instance instance)
    {

        FeatureVector fv = FeatureVector.newSparse(instance.label);

        assert (!instance.text.isEmpty());

        List<Offset> offsets = new ArrayList<Offset>();
        Set<Integer> set = new HashSet<Integer>();


        extractCharGrams(set, offsets, instance.text);

        extractWordGrams(set, offsets, instance.text);
        Collections.sort(offsets, new Comparator<Offset>()
        {
            @Override
            public int compare(Offset o1, Offset o2)
            {
                Integer x = o1.index;
                Integer y = o2.index;
                return x.compareTo(y);
            }
        });
        // Now add to fv
        for (Offset offset : offsets)
        {
            fv.add(offset);
        }

        return fv;

    }

    // When we start a new set, create a new trainer and lifecycle, and create a new cache file
    private void onNewTrainingSet() throws Exception
    {
        if (!cacheDir.exists())
        {
            log.info("Creating new cache directory: " + cacheDir);
            cacheDir.mkdirs();

        }
        boolean isAdagrad = "adagrad".equals(learningMethod);

        ModelFactory modelFactory = new LinearModelFactory(isAdagrad ? AdagradLinearModel.class : LinearModel.class);

        learner = new SGDLearner(loss, lambda, eta0, modelFactory);
        File cacheFile = File.createTempFile("sgd", ".cache", cacheDir);
        trainingRunner = new OverlappedTrainingRunner(learner);
        trainingRunner.setEpochs(epochs);
        trainingRunner.setBufferSz(bufferSz);
        trainingRunner.setLearnerUserData((int) Math.pow(2, nbits));
        trainingRunner.setCacheFile(cacheFile);
        trainingRunner.start();
    }


    public NBSVM()
    {
    }

    private void increment(Map<Integer, Long> ftable, String... words)
    {
        String joined = CollectionsManip.join(words, "_*_");
        int idx = hashFeatureEncoder.lookupOrCreate(joined);
        Long x = CollectionsManip.getOrDefault(ftable, idx, 0L);
        ftable.put(idx, x + 1L);
    }


    /**
     * Classify an instance and return the score.
     * Scores greater than 0 indicate a positive, less than 0 indicates a negative.  Beta is used to
     * determine NB weight.
     *
     * @param model The linear model
     * @param instance The training instance
     * @return A score, with less than 0 indicating a negative
     */
    public double classify(Model model, Instance instance)
    {
        FeatureVector fv = transform(instance);
        double acc = Math.log(numPositiveTrainingExamples / (double) numNegativeTrainingExamples);
        for (Offset offset : fv.getNonZeroOffsets())
        {
            acc += offset.value;
        }
        double score = model.predict(fv);

        score = beta * score + (1 - beta)*acc;
        return score;
    }

    /**
     * Evaluate a corpus of test data with a model
     * 
     * @param model The model
     * @param iterator The test corpus iterator
     * @return The accuracy of the model on this test data 
     */
    public double eval(Model model, Iterator<Instance> iterator)
    {

        int total = 0;
        int correct = 0;
        for (; iterator.hasNext(); ++total)
        {
            Instance instance = iterator.next();

            int y = instance.label;
            double fx = classify(model, instance);
            correct += (fx * y <= 0) ? 0 : 1;
        }
        System.out.println(correct + " / " + total);
        return correct / (double) total;
    }

    /**
     * Build a generative feature (word likelihood/NB) lexicon from a corpus
     * @param corpus Training data
     * @param alpha Smoothing
     */
    public void buildLexicon(Iterator<Instance> corpus, double alpha)
    {

        Map[] ftables = new Map[2];
        Map<Integer, Long> ftable0 = new HashMap<Integer, Long>();
        Map<Integer, Long> ftable1 = new HashMap<Integer, Long>();
        ftables[0] = ftable0;
        ftables[1] = ftable1;
        Long[] numTokens = new Long[]{ 0L, 0L };

        while (corpus.hasNext())
        {
            Instance instance = corpus.next();
            int labelIdx = instance.label <= 0 ? 0: 1;
            Map<Integer, Long> ftable = ftables[labelIdx];
            lexCharGrams(instance.text, ftable, numTokens, labelIdx);
            lexWordGrams(instance.text, ftable, numTokens, labelIdx);

        }

        lexicon = new HashMap<Integer, Double>();
        Set<Integer> words = new HashSet<Integer>(ftable0.keySet());
        int uniqueWordsF0 = words.size();
        Set<Integer> wordsF1 = ftable1.keySet();
        int uniqueWordsF1 = wordsF1.size();
        words.addAll(wordsF1);
        hashWordsProcessed = words.size();

        double numTotalF0 = numTokens[0] + alpha * uniqueWordsF0;
        double numTotalF1 = numTokens[1] + alpha * uniqueWordsF1;

        for (Integer word : words)
        {
            double f0 = (CollectionsManip.getOrDefault(ftable0, word, 0L) + alpha)/numTotalF0;
            double f1 = (CollectionsManip.getOrDefault(ftable1, word, 0L) + alpha)/numTotalF1;
            lexicon.put(word, Math.log(f1 / f0));
        }


    }

    public static class Params
    {

        @Parameter(description = "Lexicon corpus", names = {"--lex", "-lex"})
        public String lex;

        @Parameter(description = "Training file", names = {"--train", "-t"}, required = true)
        public String train;

        @Parameter(description = "Testing file", names = {"--eval", "-e"}, required = true)
        public String eval;

        @Parameter(description = "Model to write out", names = {"--model", "-s"})
        public String model;

        @Parameter(description = "Loss function", names = {"--loss", "-l"})
        public String loss = "hinge";

        @Parameter(description = "lambda", names = {"--lambda", "-lambda"})
        public Double lambda = 1e-5;

        @Parameter(description = "eta0, if not set, try and preprocess to find", names = {"--eta0", "-e0"})
        public Double eta0 = 0.5;

        @Parameter(description = "Number of epochs", names = {"--epochs", "-epochs"})
        public Integer epochs = 10;

        @Parameter(description = "Ring buffer size", names = {"--bufsz"})
        public Integer bufferSz = 16384;

        @Parameter(description = "Number of bits for feature hashing", names = {"--nbits"})
        public Integer nbits = 24;

        @Parameter(description = "N-grams", names = {"--ngrams"})
        public Integer ngrams = 3;

        @Parameter(description = "Learning method (sgd|adagrad)", names = {"--method"})
        public String method = "sgd";

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

    }

    /**
     * Iterator that reads from a file and makes a corpus instance available to the user
     * Data should be in form y<tab>word1 word2 ...
     */
    public static class FileIterator implements Iterator<Instance>
    {

        BufferedReader reader;
        Instance instance;
        int lineNumber = 0;

        int minSentenceLength;

        public FileIterator(File file, int minSentenceLength) throws IOException
        {
            this.minSentenceLength = minSentenceLength;
            reader = new BufferedReader(new FileReader(file));
            advance();
        }

        public boolean hasNext()
        {
            return instance != null;
        }

        public Instance next()
        {
            Instance last = instance;
            advance();
            return last;
        }

        public int getLineNumber()
        {
            return lineNumber;
        }
        private void advance()
        {
            try
            {
                final String line = reader.readLine();

                if (line == null)
                {
                    instance = null;
                    return;
                }
                ++lineNumber;

                instance = new Instance();
                // This appears to be much faster than
                final StringTokenizer tokenizer = new StringTokenizer(line, "\t ");

                try
                {

                    instance.label = Integer.valueOf(tokenizer.nextToken());

                    instance.text = new ArrayList<String>();
                    while (tokenizer.hasMoreTokens())
                    {
                        instance.text.add(tokenizer.nextToken());
                    }
                    if (instance.text.size() < minSentenceLength)
                    {
                        throw new NumberFormatException("");
                    }
                }
                catch (NumberFormatException numEx)
                {
                    //System.out.println("Bad line: " + lineNumber + ". Skipping...");
                    advance();
                }
            }
            catch (Exception ex)
            {
                System.out.println("Line number: " + lineNumber);
                throw new RuntimeException(ex);
            }

        }

        @Override
        public void remove()
        {

        }
    }

    public static void main(String[] args)
    {
        try
        {
            NBSVM nbsvm = new NBSVM();

            Params params = new Params();
            JCommander jc = new JCommander(params, args);
            jc.parse();

            nbsvm.setBufferSz(params.bufferSz);
            nbsvm.setCacheDir(new File(params.cacheDir));
            nbsvm.setEpochs(params.epochs);
            nbsvm.setLambda(params.lambda);
            nbsvm.setNbits(params.nbits);
            nbsvm.setNgrams(params.ngrams);
            nbsvm.setCharNGramWidth(params.charNgrams);
            nbsvm.setEta0(params.eta0);
            nbsvm.setBeta(params.beta);
            nbsvm.setLoss((params.loss.equals("lr") || params.loss.equals("log")) ? new LogLoss() : new HingeLoss());
            nbsvm.setLearningMethod(params.method);

            if (params.ngrams == 0 && params.charNgrams == 0)
            {
                throw new Exception("Must supply at least one word or char ngram");
            }

            Iterator<Instance> lexIterator = new FileIterator(new File(params.lex == null ? params.train: params.lex), params.minSentenceLength);
            Iterator<Instance> testIterator = new FileIterator(new File(params.eval), params.minSentenceLength);

            long lexStart = System.currentTimeMillis();
            nbsvm.buildLexicon(lexIterator, params.alpha);


            double lexSeconds = (System.currentTimeMillis() - lexStart) / 1000.0;
            System.out.println(String.format("%d hash words in lexicon, aggregated in %.02fs.  Starting training",
                    nbsvm.getHashWordsProcessed(), lexSeconds));

            long trainStart = System.currentTimeMillis();
            Iterator<Instance> trainingIterator = new FileIterator(new File(params.train), params.minSentenceLength);
            Model model = nbsvm.train(trainingIterator);

            double trainSeconds = (System.currentTimeMillis() - trainStart) / 1000.0;

            System.out.println(String.format("Trained model in %.02fs.  %d training examples seen",
                    trainSeconds, nbsvm.getNumTrainingExamples()));

            if (params.model != null)
            {
                model.save(new File(params.model));
            }


            System.out.println(String.format("Trained model %.02f seconds", trainSeconds));

            double acc = nbsvm.eval(model, testIterator);

            System.out.println(String.format("Model accuracy %.02f %%", acc * 100));
            System.out.println("Total hashing collisions " + nbsvm.getCollisions());


        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
    }
}
