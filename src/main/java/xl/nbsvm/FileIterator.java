package xl.nbsvm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.StringTokenizer;

/**
 * Iterator that reads from a file and makes a corpus instance available to the user
 * Data should be in form y<tab>word1 word2 ...
 */
public class FileIterator implements Iterator<Instance>
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

