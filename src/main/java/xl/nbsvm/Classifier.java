package xl.nbsvm;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.sgdtk.*;
import org.sgdtk.exec.OverlappedTrainingRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;

public class Classifier
{

    private Map<Integer, Double> lexicon;

    private int collisions = 0;
    private long hashWordsProcessed = 0;
    private long numPositiveTrainingExamples = 0;
    private long numNegativeTrainingExamples = 0;
    private int ngrams;
    private int nbits;
    private boolean useNBFeature;
    private boolean lowerCase;
    private boolean doClean;
    private double beta;
    private int charNGramWidth = 0;

    private HashFeatureEncoder hashFeatureEncoder;

    private static final Logger log = LoggerFactory.getLogger(Classifier.class);

            // Look up the generative feature

    public int getCollisions()
    {
        return collisions;
    }
    public Classifier(int nbits, int ngrams, int charNgrams, double beta, boolean lowerCase, boolean clean, boolean useNBFeature)
    {
        this.nbits = nbits;
        this.ngrams = ngrams;
        this.charNGramWidth = charNgrams;
        this.beta = beta;
        this.lowerCase = lowerCase;
        this.useNBFeature = useNBFeature;
        this.hashFeatureEncoder = new HashFeatureEncoder(nbits);
        this.doClean = clean;
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


    private String clean(String tok)
    {
        //String cleaned = tok.replaceAll("\"", "").replaceAll("'", "").replaceAll("`", "").replaceAll(",", "");
        String cleaned = tok;
        if (lowerCase)
        {
            cleaned = cleaned.toLowerCase();
        }
        return cleaned;
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
            t = clean(text.get(i));
            if (t.isEmpty())
                continue;

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
            t = clean(text.get(i));
            if (t.isEmpty())
                continue;

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

        if (useNBFeature)
        {
            double nbFeature = naiveBayes(fv, 0.5);
            fv.add(new Offset(0, nbFeature));
        }
        fv.getX().organize();

        return fv;

    }

    public void fit(Loss loss, int epochs, int bufferSz, double lambda, double eta0, Iterator<Instance> lexIterator, Iterator<Instance> corpus, double alpha, boolean noNBSVM, boolean useAdagrad, File cacheDir) throws Exception
    {

        SGDLearner learner;
        OverlappedTrainingRunner trainingRunner;


        long lexStart = System.currentTimeMillis();
        buildLexicon(lexIterator, alpha, noNBSVM);


        double lexSeconds = (System.currentTimeMillis() - lexStart) / 1000.0;
        System.out.println(String.format("%d hash words in lexicon, aggregated in %.02fs.  Starting training",
                getHashWordsProcessed(), lexSeconds));

        long trainStart = System.currentTimeMillis();


        if (!cacheDir.exists())
        {
            log.info("Creating new cache directory: " + cacheDir);
            cacheDir.mkdirs();

        }

        ModelFactory modelFactory = new LinearModelFactory(useAdagrad ? AdagradLinearModel.class : LinearModel.class);

        learner = new SGDLearner(loss, lambda, eta0, modelFactory);
        File cacheFile = File.createTempFile("sgd", ".cache", cacheDir);
        trainingRunner = new OverlappedTrainingRunner(learner);
        trainingRunner.setEpochs(epochs);
        trainingRunner.setBufferSz(bufferSz);
        trainingRunner.setLearnerUserData((int) Math.pow(2, nbits));
        trainingRunner.setCacheFile(cacheFile);
        trainingRunner.start();
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

            FeatureVector fv = transform(instance);

            if (fv == null)
            {
                continue;
            }
            // Add to training set asynchronously
            trainingRunner.add(fv);
        }

        // Wait until training is done
        Model model = trainingRunner.finish();

        double trainSeconds = (System.currentTimeMillis() - trainStart) / 1000.0;

        System.out.println(String.format("Trained model in %.02fs.  %d training examples seen",
                trainSeconds, getNumTrainingExamples()));


        this.model = model;
    }

    Model model;

    private void increment(Map<Integer, Long> ftable, String... words)
    {
        String joined = CollectionsManip.join(words, "_*_");
        int idx = hashFeatureEncoder.lookupOrCreate(joined);
        Long x = CollectionsManip.getOrDefault(ftable, idx, 0L);
        ftable.put(idx, x + 1L);
    }

    double naiveBayes(FeatureVector fv, double prior)
    {
        double acc = prior;
        for (Offset offset : fv.getNonZeroOffsets())
        {
            acc += offset.value;
        }
        return acc;
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
        double acc = beta < 1 ? naiveBayes(fv, Math.log(numPositiveTrainingExamples / (double) numNegativeTrainingExamples)): 0;
        double score = model.predict(fv);

        score = beta * score + (1 - beta)*acc;
        return score;
    }

    /**
     * Evaluate a corpus of test data with a model
     *
     * @param iterator The test corpus iterator
     * @return The accuracy of the model on this test data
     */
    public double eval(Iterator<Instance> iterator)
    {

        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;
        for (; iterator.hasNext(); )
        {
            Instance instance = iterator.next();

            int y = instance.label;
            double fx = classify(model, instance);

            if (fx * y <= 0)
            {
                if (y == 1)
                {
                    fn++;
                }
                else
                {
                    fp++;
                }
            }
            else if (y == 1)
            {
                tp++;
            }
            else
            {
                tn++;
            }
        }
        int correct = tn + tp;
        int total = tn + tp + fn + fp;
        System.out.println(correct + " / " + total);
        System.out.println("TP: " + tp);
        System.out.println("TN: " + tn);
        System.out.println("FP: " + fp);
        System.out.println("FN: " + fn);
        System.out.println("F1: " + fbscore(tp, fp, fn, 1));
        return correct / (double) total;
    }

    double fbscore(int truePositives, int falsePositives, int falseNegatives, int b)
    {
        double b2 = b*b;

        double num = (1 + b2) * truePositives;

        return num / (num + b2 * falseNegatives + falsePositives);
    }

    /**
     * Build a generative feature (word likelihood/NB) lexicon from a corpus
     * @param corpus Training data
     * @param alpha Smoothing
     * @param noNBSVM This strange attribute allows us to simply create a standard SVM, not NBSVM
     *                In case you are trying to compare NBSVM against SVM, for example
     */
    public void buildLexicon(Iterator<Instance> corpus, double alpha, boolean noNBSVM)
    {

        if (noNBSVM)
        {
            log.info("Using regular SVM/LR");
        }
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

        log.info("Feature extraction from corpus completed.  Beginning lexicon generation");
        lexicon = new HashMap<Integer, Double>();

        // Words is going to contain all the words in our lexicon!
        Set<Integer> words = new HashSet<Integer>(ftable0.keySet());
        int uniqueWordsF0 = words.size();
        Set<Integer> wordsF1 = ftable1.keySet();
        int uniqueWordsF1 = wordsF1.size();
        words.addAll(wordsF1);


        hashWordsProcessed = words.size();

        double numTotalF0 = numTokens[0] + alpha * uniqueWordsF0;
        double numTotalF1 = numTokens[1] + alpha * uniqueWordsF1;

        // You would only do this if you are exploiting this code to give you back and plain SVM/LR,
        // usually because you want to compare NBSVM performance against the equivalent SVM
        if (noNBSVM)
        {
            for (Integer word : words)
            {
                lexicon.put(word, 1.0);
            }
            return;
        }
        for (Integer word : words)
        {
            double f0 = (CollectionsManip.getOrDefault(ftable0, word, 0L) + alpha)/numTotalF0;
            double f1 = (CollectionsManip.getOrDefault(ftable1, word, 0L) + alpha)/numTotalF1;
            lexicon.put(word, Math.log(f1 / f0));
        }


    }

    public void save(String modelName) throws Exception
    {
        String lexiconOutput = modelName + ".lex";
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.writeValue(new File(lexiconOutput), lexicon);
        model.save(new File(modelName));
    }

    public void load(String modelName) throws Exception
    {
        String lexiconOutput = modelName + ".lex";
        ObjectMapper objectMapper = new ObjectMapper();
        lexicon = objectMapper.readValue(new File(lexiconOutput), Map.class);
    }
}
