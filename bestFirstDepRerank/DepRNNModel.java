package bestFirstDepRerank;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.util.TwoDimensionalMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdaGrad;
import reranker.Config;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by zhouh on 16-1-10.
 *
 */
public class DepRNNModel implements Serializable {

    private static final long serialVersionUID = 1;
    public HashSet<String> vocabulary = null;
    private int dim;
    Distribution distribution;
    boolean bTrain = true;


    public void setVocabulary(HashSet<String> vocabulary) {
        this.vocabulary = vocabulary;
    }

    public String getVocabWord(String word) {
        String lowerCaseWord = word.toLowerCase();
        if (vocabulary.contains(lowerCaseWord))
            return lowerCaseWord;
        return Config.UNKNOWN;
    }

    public void readPreTrain(String preTrainFile, int embedSize) throws IOException {

        BufferedReader reader = IOUtils.readerFromString(preTrainFile);
        int vocabSize = vocabulary.size();
        int preTrainedWordInVocabSize = 0;

        for (String line : IOUtils.getLineIterable(reader, false)) {
            String[] tokens = line.split("\\s{1,}");
            String caseWord = tokens[0].trim();
            String word = caseWord.toLowerCase();
            if (vocabulary.contains(word)) {

                preTrainedWordInVocabSize++;

                double[] wordEmbs = new double[embedSize];
                for (int i = 0; i < wordEmbs.length; i++)
                    wordEmbs[i] = Double.valueOf(tokens[i + 1]);

                INDArray wordEmb = Nd4j.create(wordEmbs, new int[]{embedSize, 1});
                wordVectors.put(caseWord, wordEmb);
            }
        }

        System.err.println("#####################");
        System.err.println("Pre train Word Embedding Done!");
        System.err.println("Vocab Size : " + vocabSize + ", Shot PreTrain Size : " + preTrainedWordInVocabSize + " (" + new DecimalFormat("00.00").format(((double) preTrainedWordInVocabSize / vocabSize)) + ")");

    }

    public void setBeTrain(boolean bTrain) {
        this.bTrain = bTrain;
    }

    // weight matrices from children to parent
    public TwoDimensionalMap<String, String, INDArray> binaryTransformLeft;

    // score matrices for each node type
    public TwoDimensionalMap<String, String, INDArray> binaryScoreLayerLeft;

    // weight matrices from children to parent
    public TwoDimensionalMap<String, String, INDArray> binaryTransformRight;

    // score matrices for each node type
    public TwoDimensionalMap<String, String, INDArray> binaryScoreLayerRight;

    // word representations
    public Map<String, INDArray> wordVectors;

    public DepRNNModel(int dim) {
        this.dim = dim;
        binaryTransformLeft = new TwoDimensionalMap<>();
        binaryScoreLayerLeft = new TwoDimensionalMap<>();
        binaryTransformRight = new TwoDimensionalMap<>();
        binaryScoreLayerRight = new TwoDimensionalMap<>();
        wordVectors = new HashMap<>();

        distribution = new UniformDistribution(-0.001, 0.001);
    }

    public DepRNNModel() {
        distribution = new UniformDistribution(-0.001, 0.001);
    }

    /**
     * Insert binary transform according to left and right label
     *
     * @param label1
     * @param label2
     * @param bLeft
     * @param bRandom
     * @return
     */
    public INDArray insertBinaryTransform(String label1, String label2, boolean bLeft, boolean bRandom) {
        INDArray retval = null;
        if (bLeft) {
            if (!binaryTransformLeft.contains(label1, label2)) {
                if (bRandom)
                    retval = Nd4j.rand(new int[]{dim, 2 * dim + 1}, distribution);
                else
                    retval = Nd4j.zeros(dim, 2 * dim + 1);

                binaryTransformLeft.put(label1, label2, retval);

            }

        } else {
            if (!binaryTransformRight.contains(label1, label2)) {
                if (bRandom)
                    retval = Nd4j.rand(new int[]{dim, 2 * dim + 1}, distribution);
                else
                    retval = Nd4j.zeros(dim, 2 * dim + 1);

                binaryTransformRight.put(label1, label2, retval);

            }

        }
        return retval;
    }

    /**
     * Insert binary score layer according to left and right label
     *
     * @param label1
     * @param label2
     * @param bLeft
     * @param bRandom
     * @return
     */
    public INDArray insertBinaryScoreLayer(String label1, String label2, boolean bLeft, boolean bRandom) {
        INDArray retval = null;

        if (bLeft) {
            if (!binaryScoreLayerLeft.contains(label1, label2)) {
                if (bRandom)
                    retval = Nd4j.rand(new int[]{1, dim}, distribution);
                else
                    retval = Nd4j.zeros(1, dim);
                binaryScoreLayerLeft.put(label1, label2, retval);
            }

        } else {
            if (!binaryScoreLayerRight.contains(label1, label2)) {
                if (bRandom)
                    retval = Nd4j.rand(new int[]{1, dim}, distribution);
                else
                    retval = Nd4j.zeros(1, dim);
                binaryScoreLayerRight.put(label1, label2, retval);
            }
        }


        return retval;

    }

    public INDArray insertWordVector(String word, boolean bRandom) {
        INDArray retval = null;

        word = getVocabWord(word);

        if (!wordVectors.containsKey(word)) {
            if (bRandom)
                retval = Nd4j.rand(new int[]{dim, 1}, distribution);
            else
                retval = Nd4j.zeros(dim, 1);
            wordVectors.put(word, retval);
        }

        return retval;

    }

    /**
     * insert the unk word and rules into this model
     * @param word
     * @param bRandom
     */
    public void insertUNKWord(String word, boolean bRandom) {
        INDArray retval = null;

        if (!wordVectors.containsKey(word)) {
            if (bRandom)
                retval = Nd4j.rand(new int[]{dim, 1}, distribution);
            else
                retval = Nd4j.zeros(dim, 1);
            wordVectors.put(word, retval);
        }

        // insert the unk rules
        insertBinaryScoreLayer(word, word, true, true);
        insertBinaryScoreLayer(word, word, false, true);
        insertBinaryTransform(word, word, false, true);
        insertBinaryTransform(word, word, true, true);


    }

    public INDArray getBinaryTransform(DepRerankingTree tree) {

        if (tree.numChildren() == 0)
            throw new RuntimeException("Bad tree for Binary Rule!");

        if (tree.bLeft) {
            if (binaryTransformLeft.contains(
                    tree.getChild(0).getLabel(),
                    tree.getChild(1).getLabel())) {

//            System.err.println("Found rule : "+ tree.getChild(0).getLabel() + " "+tree.getChild(1).getLabel());

                return binaryTransformLeft.get(
                        tree.getChild(0).getLabel(),
                        tree.getChild(1).getLabel());
            } else {
                return binaryTransformLeft.get(Config.UNKNOWN, Config.UNKNOWN);

            }

        } else {
            if (binaryTransformRight.contains(
                    tree.getChild(0).getLabel(),
                    tree.getChild(1).getLabel())) {


                return binaryTransformRight.get(
                        tree.getChild(0).getLabel(),
                        tree.getChild(1).getLabel());
            } else {
                return binaryTransformRight.get(Config.UNKNOWN, Config.UNKNOWN);

            }

        }
    }

    public INDArray getBinaryScoreLayer(DepRerankingTree tree) {
        if (tree.numChildren() == 0)
            throw new RuntimeException("Bad tree for Binary Rule!");

        if (tree.bLeft) {
            if (binaryScoreLayerLeft.contains(
                    tree.getChild(0).getLabel(),
                    tree.getChild(1).getLabel())) {

//            System.err.println("Found rule : "+ tree.getChild(0).getLabel() + " "+tree.getChild(1).getLabel());

                return binaryScoreLayerLeft.get(
                        tree.getChild(0).getLabel(),
                        tree.getChild(1).getLabel());
            } else {
                return binaryScoreLayerLeft.get(Config.UNKNOWN, Config.UNKNOWN);

            }

        } else {
            if (binaryScoreLayerRight.contains(
                    tree.getChild(0).getLabel(),
                    tree.getChild(1).getLabel())) {


                return binaryScoreLayerRight.get(
                        tree.getChild(0).getLabel(),
                        tree.getChild(1).getLabel());
            } else {
                return binaryScoreLayerRight.get(Config.UNKNOWN, Config.UNKNOWN);

            }

        }
    }

    /**
     * return the embedding of a word, after tanh activation
     *
     * @param tree
     * @return
     */
    public INDArray getWordVector(DepRerankingTree tree) {
        String word = getVocabWord(tree.getWord());
        if (wordVectors.containsKey(word))
            return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", wordVectors.get(word).dup()));
        else
            return null;
    }


    public INDArray getOrInsertBinaryTransform(DepRerankingTree tree) {

        if (tree.bLeft) {
            if (binaryTransformLeft.contains(
                    tree.getChild(0).getLabel(),
                    tree.getChild(1).getLabel()))
                return binaryTransformLeft.get(
                        tree.getChild(0).getLabel(),
                        tree.getChild(1).getLabel());
            else {

                return insertBinaryTransform(tree.getChild(0).getLabel(), tree.getChild(1).getLabel(), tree.bLeft, false);
            }

        } else {
            if (binaryTransformRight.contains(
                    tree.getChild(0).getLabel(),
                    tree.getChild(1).getLabel()))
                return binaryTransformRight.get(
                        tree.getChild(0).getLabel(),
                        tree.getChild(1).getLabel());
            else
                return insertBinaryTransform(tree.getChild(0).getLabel(), tree.getChild(1).getLabel(), tree.bLeft, false);

        }
    }

    public INDArray getOrInsertBinaryScoreLayer(DepRerankingTree tree) {

        if (tree.bLeft) {
            if (binaryScoreLayerLeft.contains(
                    tree.getChild(0).getLabel(),
                    tree.getChild(1).getLabel()))
                return binaryScoreLayerLeft.get(
                        tree.getChild(0).getLabel(),
                        tree.getChild(1).getLabel());
            else {

                return insertBinaryScoreLayer(tree.getChild(0).getLabel(), tree.getChild(1).getLabel(), tree.bLeft, false);
            }

        } else {
            if (binaryScoreLayerRight.contains(
                    tree.getChild(0).getLabel(),
                    tree.getChild(1).getLabel()))
                return binaryScoreLayerRight.get(
                        tree.getChild(0).getLabel(),
                        tree.getChild(1).getLabel());
            else
                return insertBinaryScoreLayer(tree.getChild(0).getLabel(), tree.getChild(1).getLabel(), tree.bLeft, false);

        }
    }

    public INDArray getOrInsertWordVector(DepRerankingTree tree) {
        String word = getVocabWord(tree.getWord());
        if (wordVectors.containsKey(word))
            return wordVectors.get(word);
        else
            return insertWordVector(word, false);
    }


    /**
     * TODO complete the read and write model module
     *
     * @param modelFileName
     */
    public static DepRNNModel readModel(String modelFileName) {

        DepRNNModel model = new DepRNNModel();

        ObjectInputStream ois1 = null;
        try {
            ois1 = new ObjectInputStream(new FileInputStream(modelFileName));

            System.err.print("Begin to read vocabulary ");
            model.vocabulary = (HashSet<String>) ois1.readObject();
            System.err.print("binaryTransform ");
            model.binaryTransformLeft = (TwoDimensionalMap<String, String, INDArray>) ois1.readObject();
            model.binaryTransformRight = (TwoDimensionalMap<String, String, INDArray>) ois1.readObject();
            System.err.print("binaryScoreLayer ");
            model.binaryScoreLayerLeft = (TwoDimensionalMap<String, String, INDArray>) ois1.readObject();
            model.binaryScoreLayerRight = (TwoDimensionalMap<String, String, INDArray>) ois1.readObject();
            System.err.print("wordVectors ");
            model.wordVectors = (Map<String, INDArray>) ois1.readObject();
            System.err.print("unknowWord and dim.");
            model.dim = (int) ois1.readObject();

            ois1.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        return model;
    }

    public static void writeModel(String modelFileName, DepRNNModel model) {

        System.err.println("Begin to Write Model" + modelFileName);
        try {
            ObjectOutputStream oos1 = new ObjectOutputStream(new FileOutputStream(modelFileName));
            System.err.print("Begin to write vocabulary ");
            oos1.writeObject(model.vocabulary);
            System.err.print("binaryTransform ");
            oos1.writeObject(model.binaryTransformLeft);
            oos1.writeObject(model.binaryTransformRight);
            System.err.print("binaryScoreLayer ");
            oos1.writeObject(model.binaryScoreLayerLeft);
            oos1.writeObject(model.binaryScoreLayerRight);
            System.err.print("wordVectors ");
            oos1.writeObject(model.wordVectors);
            System.err.print("unknowWord and dim.");
            oos1.writeObject(model.dim);

            oos1.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.err.println("Finish to Write Model");
    }

    /**
     * New a new model style object for store gradients of matrices in model
     *
     * @return
     */
    public DepRNNModel createGradients() {

        DepRNNModel gradient = new DepRNNModel(dim);
        gradient.setVocabulary(vocabulary);

        // insert the binary transform gradients
        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> btIterator = binaryTransformLeft.iterator();
        while (btIterator.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = btIterator.next();
            gradient.insertBinaryTransform(it.getFirstKey(), it.getSecondKey(), true, false);
        }

        // insert the binary transform score layer
        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIterator = binaryScoreLayerLeft.iterator();
        while (bslIterator.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIterator.next();
            gradient.insertBinaryScoreLayer(it.getFirstKey(), it.getSecondKey(), true, false);
        }

        // insert the binary transform gradients
        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> btIteratorR = binaryTransformRight.iterator();
        while (btIteratorR.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = btIteratorR.next();
            gradient.insertBinaryTransform(it.getFirstKey(), it.getSecondKey(), false, false);
        }

        // insert the binary transform score layer
        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIteratorR = binaryScoreLayerRight.iterator();
        while (bslIteratorR.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIteratorR.next();
            gradient.insertBinaryScoreLayer(it.getFirstKey(), it.getSecondKey(), false, false);
        }

        // insert the word vector gradients
        for (Map.Entry<String, INDArray> wvEntry : wordVectors.entrySet())
            gradient.insertWordVector(wvEntry.getKey(), false);


        return gradient;
    }


    public IdentityHashMap<INDArray, AdaGrad> createAdaGradSquare(double fBPRate) {

        IdentityHashMap<INDArray, AdaGrad> paras2SquareGradients = new IdentityHashMap<>();

        // insert the binary transform gradients
        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> btIterator = binaryTransformLeft.iterator();
        while (btIterator.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = btIterator.next();
            paras2SquareGradients.put(it.getValue(), new AdaGrad(it.getValue().shape(), fBPRate));
        }

        // insert the binary transform score layer
        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIterator = binaryScoreLayerLeft.iterator();
        while (bslIterator.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIterator.next();
            paras2SquareGradients.put(it.getValue(), new AdaGrad(it.getValue().shape(), fBPRate));
        }

        // insert the binary transform gradients
        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> btIteratorR = binaryTransformRight.iterator();
        while (btIteratorR.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = btIteratorR.next();
            paras2SquareGradients.put(it.getValue(), new AdaGrad(it.getValue().shape(), fBPRate));
        }

        // insert the binary transform score layer
        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIteratorR = binaryScoreLayerRight.iterator();
        while (bslIteratorR.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIteratorR.next();
            paras2SquareGradients.put(it.getValue(), new AdaGrad(it.getValue().shape(), fBPRate));
        }

        // insert the word vector gradients
        for (Map.Entry<String, INDArray> wvEntry : wordVectors.entrySet())
            paras2SquareGradients.put(wvEntry.getValue(), new AdaGrad(wvEntry.getValue().shape(), fBPRate));

        return paras2SquareGradients;
    }

    /**
     * update the gradients from one batch examples to the network paras
     * Using Adagrad Updating and add the l2-norm
     *
     * @param gradients
     * @param gradientSquareMap
     * @param batchSize
     * @param regRate
     */
    public void updateModel(DepRNNModel gradients, IdentityHashMap<INDArray, AdaGrad> gradientSquareMap,
                            int batchSize, double regRate) {

        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> btIteratorL = gradients.binaryTransformLeft.iterator();
        while (btIteratorL.hasNext()) {

            TwoDimensionalMap.Entry<String, String, INDArray> it = btIteratorL.next();
            INDArray toBeUpdated = binaryTransformLeft.get(it.getFirstKey(), it.getSecondKey());
            INDArray gradient = it.getValue();
            gradient.muli(1.0 / batchSize);
            gradient.addi(toBeUpdated.mul(regRate)); // add l-2 norm to gradients

            INDArray learningRates = gradientSquareMap.get(toBeUpdated).getGradient(gradient, 0);
//            gradient.muli( learningRates );
            toBeUpdated.subi(learningRates);
//            toBeUpdated.subi(gradient.muli(0.1));
            gradient.muli(0);
        }

        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIteratorL = gradients.binaryScoreLayerLeft.iterator();
        while (bslIteratorL.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIteratorL.next();

            INDArray toBeUpdated = binaryScoreLayerLeft.get(it.getFirstKey(), it.getSecondKey());
            INDArray gradient = it.getValue();
            gradient.muli(1.0 / batchSize);
            gradient.addi(toBeUpdated.mul(regRate)); // add l-2 norm to gradients

            INDArray learningRates = gradientSquareMap.get(toBeUpdated).getGradient(gradient, 0);
//            gradient.muli( learningRates );

            toBeUpdated.subi(learningRates);
//            toBeUpdated.subi(gradient.muli(0.1));
            gradient.muli(0);
        }

        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> btIteratorR = gradients.binaryTransformRight.iterator();
        while (btIteratorR.hasNext()) {

            TwoDimensionalMap.Entry<String, String, INDArray> it = btIteratorR.next();
            INDArray toBeUpdated = binaryTransformRight.get(it.getFirstKey(), it.getSecondKey());
            INDArray gradient = it.getValue();
            gradient.muli(1.0 / batchSize);
            gradient.addi(toBeUpdated.mul(regRate)); // add l-2 norm to gradients

            INDArray learningRates = gradientSquareMap.get(toBeUpdated).getGradient(gradient, 0);
//            gradient.muli( learningRates );
            toBeUpdated.subi(learningRates);
//            toBeUpdated.subi(gradient.muli(0.1));
            gradient.muli(0);
        }

        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIteratorR = gradients.binaryScoreLayerRight.iterator();
        while (bslIteratorR.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIteratorR.next();

            INDArray toBeUpdated = binaryScoreLayerRight.get(it.getFirstKey(), it.getSecondKey());
            INDArray gradient = it.getValue();
            gradient.muli(1.0 / batchSize);
            gradient.addi(toBeUpdated.mul(regRate)); // add l-2 norm to gradients

            INDArray learningRates = gradientSquareMap.get(toBeUpdated).getGradient(gradient, 0);
//            gradient.muli( learningRates );

            toBeUpdated.subi(learningRates);
//            toBeUpdated.subi(gradient.muli(0.1));
            gradient.muli(0);
        }

        // insert the word vector gradients
        for (Map.Entry<String, INDArray> wvEntry : gradients.wordVectors.entrySet()) {
            INDArray toBeUpdated = wordVectors.get(wvEntry.getKey());
            INDArray gradient = wvEntry.getValue();
            gradient.muli(1.0 / batchSize);
            gradient.addi(toBeUpdated.mul(regRate)); // add l-2 norm to gradients

            INDArray learningRates = gradientSquareMap.get(toBeUpdated).getGradient(gradient, 0);
//            gradient.muli( learningRates );
            toBeUpdated.subi(learningRates);
//            toBeUpdated.subi(gradient.muli(0.1));
            gradient.muli(0);
        }
    }

    /**
     * Merge the gradients from one sentence to the final gradients for updating paras
     *
     * @param gradients
     * @param rate
     */
    public void merge(DepRNNModel gradients, double rate) {

        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> btIteratorL = gradients.binaryTransformLeft.iterator();
        while (btIteratorL.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = btIteratorL.next();
            binaryTransformLeft.get(it.getFirstKey(), it.getSecondKey()).addi(it.getValue().muli(rate));
        }

        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIteratorL = gradients.binaryScoreLayerLeft.iterator();
        while (bslIteratorL.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIteratorL.next();
            binaryScoreLayerLeft.get(it.getFirstKey(), it.getSecondKey()).addi(it.getValue().muli(rate));
        }

        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> btIteratorR = gradients.binaryTransformRight.iterator();
        while (btIteratorR.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = btIteratorR.next();
            binaryTransformRight.get(it.getFirstKey(), it.getSecondKey()).addi(it.getValue().muli(rate));
        }

        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIteratorR = gradients.binaryScoreLayerRight.iterator();
        while (bslIteratorR.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIteratorR.next();
            binaryScoreLayerRight.get(it.getFirstKey(), it.getSecondKey()).addi(it.getValue().muli(rate));
        }

        // insert the word vector gradients
        for (Map.Entry<String, INDArray> wvEntry : gradients.wordVectors.entrySet()) {
            wordVectors.get(wvEntry.getKey()).addi(wvEntry.getValue().muli(rate));
        }
    }
}
