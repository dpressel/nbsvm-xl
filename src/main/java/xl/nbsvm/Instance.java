package xl.nbsvm;

import java.util.List;

/**
 * Simple view of a single training instance containing only
 * a label and tokenized text
 */
public class Instance
{
    int label;
    List<String> text;
}