package xl.nbsvm;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.sgdtk.*;
import org.sgdtk.exec.OverlappedTrainingLifecycle;
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
    private OverlappedTrainingLifecycle trainingLifecycle;
    private int collisions = 0;
    private long wordsProcessed = 0;
    private long numPositiveTrainingExamples = 0;
    private long numNegativeTrainingExamples = 0;
    private File cacheDir;
    private Loss loss;
    private int epochs;
    private int bufferSz;
    private double lambda;
    private Map<String, Double> lexicon;
    private int ngrams;
    private int nbits;
    private double eta0;
    private double beta;

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
    public long getWordsProcessed()
    {
        return wordsProcessed;
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
    public Model train(Iterator<Instance> corpus) throws IOException
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
            trainingLifecycle.add(fv);
        }

        // Wait until training is done
        return trainingLifecycle.finish();
    }

    private void toFeature(Set<Integer> set, List<Offset> offsets, String... str)
    {

        String joined = CollectionsManip.join(str, "_*_"); // This is borrowed from Mesnil's implementation
        int idx = hashFeatureEncoder.lookupOrCreate(joined);

        if (!set.contains(idx))
        {
            set.add(idx);
            Double v = lexicon.get(joined);
            if (v == null)
            {
                return;
            }
            offsets.add(new Offset(idx, v == null ? 0. : v));
        }
        else
        {
            collisions++;
        }
    }

    // instance should be tabulated already so just hash the keys and set the values
    private FeatureVector transform(Instance instance)
    {

        FeatureVector fv = FeatureVector.newSparse(instance.label);

        assert (!instance.text.isEmpty());

        List<Offset> offsets = new ArrayList<Offset>();
        Set<Integer> set = new HashSet<Integer>();

        // Up to 4-grams, nobody needs more than that
        String t = null;
        String l = null;
        String ll = null;
        String lll;

        for (int i = 0, sz = instance.text.size(); i < sz; ++i)
        {
            // Circular
            lll = ll;
            ll = l;
            l = t;
            t = instance.text.get(i);
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


    private void onNewTrainingSet() throws IOException
    {
        if (!cacheDir.exists())
        {
            log.info("Creating new cache directory: " + cacheDir);
            cacheDir.mkdirs();

        }
        learner = new SGDLearner(loss, lambda, eta0);
        File cacheFile = File.createTempFile("sgd", ".cache", cacheDir);
        trainingLifecycle = new OverlappedTrainingLifecycle(epochs, bufferSz, learner, (int) Math.pow(2, nbits), cacheFile);
    }


    public NBSVM()
    {
    }

    private static void increment(Map<String, Integer> ftable, String... words)
    {
        String joined = CollectionsManip.join(words, "_*_");
        Integer x = CollectionsManip.getOrDefault(ftable, joined, 0);
        ftable.put(joined, x + 1);
        ftable.put("N_TOTAL_WORDS", CollectionsManip.getOrDefault(ftable, "N_TOTAL_WORDS", 0) + 1);
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
        ftables[0] = new HashMap<String, Integer>();
        ftables[1] = new HashMap<String, Integer>();
        while (corpus.hasNext())
        {
            Instance instance = corpus.next();

            Map<String, Integer> ftable = ftables[instance.label <= 0 ? 0: 1];

            // Up to 4-grams, nobody needs more than that
            String t = null;
            String l = null;
            String ll = null;
            String lll;

            for (int i = 0, sz = instance.text.size(); i < sz; ++i)
            {
                // Circular
                lll = ll;
                ll = l;
                l = t;
                t = instance.text.get(i);
                // unigram
                increment(ftable, t);

                // bigram?
                if (ngrams > 1 && l != null)
                {
                    // trigram?
                    increment(ftable, l, t);
                    if (ngrams > 2 && ll != null)
                    {
                        increment(ftable, ll, l, t);
                    }
                    // 4-gram?
                    if (ngrams > 3 && lll != null)
                    {
                        increment(ftable, lll, ll, l, t);
                    }
                }
            }
        }
        lexicon = new HashMap<String, Double>();
        Set<String> words = new HashSet<String>(ftables[0].keySet());
        int uniqueWordsF0 = words.size();
        Set<String> wordsF1 = ftables[1].keySet();
        int uniqueWordsF1 = wordsF1.size();
        words.addAll(wordsF1);
        wordsProcessed = words.size();
        Map<String, Integer> ftable0 = ftables[0];
        Map<String, Integer> ftable1 = ftables[1];


        double numTotalF0 = ftable0.get("N_TOTAL_WORDS") + alpha * uniqueWordsF0;
        double numTotalF1 = ftable1.get("N_TOTAL_WORDS") + alpha * uniqueWordsF1;

        for (String word : words)
        {
            double f0 = (CollectionsManip.getOrDefault(ftable0, word, 0) + alpha)/numTotalF0;
            double f1 = (CollectionsManip.getOrDefault(ftable1, word, 0) + alpha)/numTotalF1;
            lexicon.put(word, Math.log(f1 / f0));
        }


    }

    public static class Params
    {

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
        public Integer epochs = 3;

        @Parameter(description = "Ring buffer size", names = {"--bufsz"})
        public Integer bufferSz = 16384;

        @Parameter(description = "Number of bits for feature hashing", names = {"--nbits"})
        public Integer nbits = 24;

        @Parameter(description = "N-grams", names = {"--ngrams"})
        public Integer ngrams = 3;

        @Parameter(description = "SVM to Naive bayes interpolation factor (1 means all SVM, 0 all NB)", names = {"--beta"})
        public Double beta = 0.95;

        @Parameter(description = "Control generative feature smoothing", names = {"--alpha"})
        public Double alpha = 1.0;

        @Parameter(description = "Cache dir", names = {"--cache-dir"})
        public String cacheDir = "cache";

    }

    /**
     * Iterator that reads from a file and makes a corpus instance available to the user
     * Data should be in form y<tab>word1 word2 ...
     */
    public static class FileIterator implements Iterator<Instance>
    {

        BufferedReader reader;
        Instance instance;

        public FileIterator(File file) throws IOException
        {
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

                instance = new Instance();
                // This appears to be much faster than
                final StringTokenizer tokenizer = new StringTokenizer(line, "\t ");

                instance.label = Integer.valueOf(tokenizer.nextToken());

                instance.text = new ArrayList<String>();
                while (tokenizer.hasMoreTokens())
                {
                    instance.text.add(tokenizer.nextToken());
                }
            }
            catch (IOException ioEx)
            {
                throw new RuntimeException(ioEx);
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
            nbsvm.setEta0(params.eta0);
            nbsvm.setBeta(params.beta);
            nbsvm.setLoss(params.loss.equals("lr") ? new LogLoss() : new HingeLoss());


            Iterator<Instance> trainingIterator = new FileIterator(new File(params.train));
            Iterator<Instance> testIterator = new FileIterator(new File(params.eval));

            long lexStart = System.currentTimeMillis();
            nbsvm.buildLexicon(trainingIterator, params.alpha);

            double lexSeconds = (System.currentTimeMillis() - lexStart) / 1000.0;
            System.out.println(String.format("%d words in lexicon, aggregated in %.02fs.  Starting training",
                    nbsvm.getWordsProcessed(), lexSeconds));

            long trainStart = System.currentTimeMillis();
            trainingIterator = new FileIterator(new File(params.train));
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

            System.out.println(String.format("Model accuracy %.02f %%", acc));
            System.out.println("Total hashing collsions " + nbsvm.getCollisions());


        }
        catch (IOException ioEx)
        {
            ioEx.printStackTrace();
        }
    }
}
