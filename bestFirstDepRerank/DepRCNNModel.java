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

public class DepRCNNModel implements Serializable {

    private static final long serialVersionUID = 1;
    public HashSet<String> vocabulary = null;
    public HashSet<String> disVocabulary = null;
    private int dim;
    Distribution distribution;
    boolean bTrain = true;


    public void setVocabulary(HashSet<String> vocabulary, HashSet<String> disVocabulary) {
        this.vocabulary = vocabulary;
        this.disVocabulary = disVocabulary;
    }

//    public void setDisVocabulary(HashSet<String> disVocabulary) {
//        this.disVocabulary = disVocabulary;
//    }

    public String getVocabWord(String word) {
        String lowerCaseWord = word.toLowerCase();
        if (vocabulary.contains(lowerCaseWord))
            return lowerCaseWord;
        else{
//            System.err.println("UNK Word: " + word);
            return Config.UNKNOWN;
        }
    }

    public String getVocabDis(String dis) {
        if (disVocabulary.contains(dis))
            return dis;
        else{

            System.err.println("UNK dis: " + dis);
            if(dis.startsWith("L")) {
                return Config.UNKNOWN+"/L";
            }
            else {
                return Config.UNKNOWN + "/R";
            }
        }
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

    public INDArray getNonlinearEmb(String word){
        String vocabWord = getVocabWord(word);
        return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", wordVectors.get(vocabWord).dup()));
    }

    public void setBeTrain(boolean bTrain) {
        this.bTrain = bTrain;
    }

    // weight matrices from children to parent
    public TwoDimensionalMap<String, String, INDArray> binaryTransformLeft;

    // score matrices for each node type
    public TwoDimensionalMap<String, String, INDArray> binaryScoreLayerLeft;

    // word representations
    public Map<String, INDArray> wordVectors;

    public Map<String, INDArray> disVectors;

    public DepRCNNModel(int dim) {
        this.dim = dim;
        binaryTransformLeft = new TwoDimensionalMap<>();
        binaryScoreLayerLeft = new TwoDimensionalMap<>();
        wordVectors = new HashMap<>();
        disVectors = new HashMap<>();

        distribution = new UniformDistribution(-0.001, 0.001);
    }

    public DepRCNNModel() {
        distribution = new UniformDistribution(-0.001, 0.001);
    }

    /**
     * Insert binary transform according to left and right label
     *
     * @param label1
     * @param label2
     * @param bRandom
     * @return
     */
    public INDArray insertBinaryTransform(String label1, String label2, boolean bRandom) {
        INDArray retval = null;
            if (!binaryTransformLeft.contains(label1, label2)) {
                if (bRandom)
                    retval = Nd4j.rand(new int[]{dim, DepRCNN.COMPOSE_NUM * dim}, distribution);
                else
                    retval = Nd4j.zeros(dim, DepRCNN.COMPOSE_NUM * dim);

                binaryTransformLeft.put(label1, label2, retval);

            }

        return retval;
    }

    /**
     * Insert binary score layer according to left and right label
     *
     * @param label1
     * @param label2
     * @param bRandom
     * @return
     */
    public INDArray insertBinaryScoreLayer(String label1, String label2, boolean bRandom) {
        INDArray retval = null;

            if (!binaryScoreLayerLeft.contains(label1, label2)) {
                if (bRandom)
                    retval = Nd4j.rand(new int[]{1, dim}, distribution);
                else
                    retval = Nd4j.zeros(1, dim);
                binaryScoreLayerLeft.put(label1, label2, retval);
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

    public INDArray insertDisVector(String disInfo, boolean bRandom) {
        INDArray retval = null;

        disInfo = getVocabDis(disInfo);

        if (!disVectors.containsKey(disInfo)) {
            if (bRandom)
                retval = Nd4j.rand(new int[]{dim, 1}, distribution);
            else
                retval = Nd4j.zeros(dim, 1);
            disVectors.put(disInfo, retval);
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
        insertBinaryScoreLayer(word, word,  true);
        insertBinaryTransform(word, word,  true);

    }

    public void insertUNKDis(){

        INDArray retval = Nd4j.rand(new int[]{dim, 1}, distribution);
        disVectors.put(Config.UNKNOWN + "/L", retval);
        disVocabulary.add(Config.UNKNOWN + "/L");
        INDArray retval2 = Nd4j.rand(new int[]{dim, 1}, distribution);
        disVectors.put(Config.UNKNOWN + "/R", retval2);
        disVocabulary.add(Config.UNKNOWN + "/R");

    }

    public INDArray getBinaryTransform(String headPos, String childPos) {

            if (binaryTransformLeft.contains(
                    headPos,
                    childPos)) {

//            System.err.println("Found rule : "+ tree.getChild(0).getLabel() + " "+tree.getChild(1).getLabel());

                return binaryTransformLeft.get(
                        headPos,
                        childPos);
            } else {
                return binaryTransformLeft.get(Config.UNKNOWN, Config.UNKNOWN);

            }

    }

    public INDArray getBinaryScoreLayer(String headPos, String childPos) {

            if (binaryScoreLayerLeft.contains(
                    headPos,
                    childPos)) {

//            System.err.println("Found rule : "+ tree.getChild(0).getLabel() + " "+tree.getChild(1).getLabel());

                return binaryScoreLayerLeft.get(
                        headPos,
                        childPos);
            } else {
                return binaryScoreLayerLeft.get(Config.UNKNOWN, Config.UNKNOWN);

            }

    }

//    public INDArray getWordVector(DepRCNNTree tree) {
//        String word = getVocabWord(tree.getWord());
//        if (wordVectors.containsKey(word))
//            return wordVectors.get(word);
//        else
//            return null;
//    }

    public INDArray getDisVector(String dis) {
        dis = getVocabDis(dis);
        if (disVectors.containsKey(dis))
            return disVectors.get(dis);
        else{
            System.err.println("Return UNK Dis!" + dis);
            return null;


        }
    }


    public INDArray getOrInsertBinaryTransform(String headPos, String childPos) {

            if (binaryTransformLeft.contains(
                    headPos,
                    childPos))
                return binaryTransformLeft.get(
                        headPos,
                        childPos);
            else {

                return insertBinaryTransform(headPos, childPos, false);
            }

    }

    public INDArray getOrInsertBinaryScoreLayer(String headPos, String childPos) {

            if (binaryScoreLayerLeft.contains(
                    headPos,
                    childPos))
                return binaryScoreLayerLeft.get(
                        headPos,
                        childPos);
            else {

                return insertBinaryScoreLayer(headPos, childPos, false);
            }

    }

    public INDArray getOrInsertWordVector(String word) {
        String vocWord = getVocabWord(word);
        if (wordVectors.containsKey(vocWord))
            return wordVectors.get(vocWord);
        else
            return insertWordVector(vocWord, false);
    }

    public INDArray getOrInsertDisVector(String dis) {
        dis = getVocabDis(dis);
        if (disVectors.containsKey(dis))
            return disVectors.get(dis);
        else
            return insertDisVector(dis, false);
    }


    /**
     * TODO complete the read and write model module
     *
     * @param modelFileName
     */
    public static DepRCNNModel readModel(String modelFileName) {

        DepRCNNModel model = new DepRCNNModel();

        ObjectInputStream ois1 = null;
        try {
            ois1 = new ObjectInputStream(new FileInputStream(modelFileName));

            System.err.print("Begin to read vocabulary ");
            model.vocabulary = (HashSet<String>) ois1.readObject();
            System.err.print("dis vocabulary ");
            model.disVocabulary = (HashSet<String>) ois1.readObject();
            System.err.print("binaryTransform ");
            model.binaryTransformLeft = (TwoDimensionalMap<String, String, INDArray>) ois1.readObject();
            System.err.print("binaryScoreLayer ");
            model.binaryScoreLayerLeft = (TwoDimensionalMap<String, String, INDArray>) ois1.readObject();
            System.err.print("wordVectors ");
            model.wordVectors = (Map<String, INDArray>) ois1.readObject();
            System.err.print("disVectors ");
            model.disVectors = (Map<String, INDArray>) ois1.readObject();
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

    public static void writeModel(String modelFileName, DepRCNNModel model) {

        System.err.println("Begin to Write Model" + modelFileName);
        try {
            ObjectOutputStream oos1 = new ObjectOutputStream(new FileOutputStream(modelFileName));
            System.err.print("Begin to write vocabulary ");
            oos1.writeObject(model.vocabulary);
            System.err.print("Begin to write dis dis vocabulary ");
            oos1.writeObject(model.disVocabulary);
            System.err.print("binaryTransform ");
            oos1.writeObject(model.binaryTransformLeft);
            System.err.print("binaryScoreLayer ");
            oos1.writeObject(model.binaryScoreLayerLeft);
            System.err.print("wordVectors ");
            oos1.writeObject(model.wordVectors);
            System.err.print("disVectors ");
            oos1.writeObject(model.disVectors);
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
    public DepRCNNModel createGradients() {

        DepRCNNModel gradient = new DepRCNNModel(dim);
        gradient.setVocabulary(vocabulary, disVocabulary);

        // insert the binary transform gradients
        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> btIterator = binaryTransformLeft.iterator();
        while (btIterator.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = btIterator.next();
            gradient.insertBinaryTransform(it.getFirstKey(), it.getSecondKey(), false);
        }

        // insert the binary transform score layer
        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIterator = binaryScoreLayerLeft.iterator();
        while (bslIterator.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIterator.next();
            gradient.insertBinaryScoreLayer(it.getFirstKey(), it.getSecondKey(), false);
        }

        // insert the word vector gradients
        for (Map.Entry<String, INDArray> wvEntry : wordVectors.entrySet())
            gradient.insertWordVector(wvEntry.getKey(), false);

        // insert the word vector gradients
        for (Map.Entry<String, INDArray> disVEntry : disVectors.entrySet())
            gradient.insertDisVector(disVEntry.getKey(), false);


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

        // insert the word vector gradients
        for (Map.Entry<String, INDArray> wvEntry : wordVectors.entrySet())
            paras2SquareGradients.put(wvEntry.getValue(), new AdaGrad(wvEntry.getValue().shape(), fBPRate));

        // insert the dis vector gradients
        for (Map.Entry<String, INDArray> disVEntry : disVectors.entrySet())
            paras2SquareGradients.put(disVEntry.getValue(), new AdaGrad(disVEntry.getValue().shape(), fBPRate));


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
    public void updateModel(DepRCNNModel gradients, IdentityHashMap<INDArray, AdaGrad> gradientSquareMap,
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

        // insert the word vector gradients
        for (Map.Entry<String, INDArray> disVEntry : gradients.disVectors.entrySet()) {
            INDArray toBeUpdated = disVectors.get(disVEntry.getKey());
            INDArray gradient = disVEntry.getValue();
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
    public void merge(DepRCNNModel gradients, double rate) {

        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> btIteratorL = gradients.binaryTransformLeft.iterator();
        while (btIteratorL.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = btIteratorL.next();
            binaryTransformLeft.get(it.getFirstKey(), it.getSecondKey()).addi(it.getValue().muli(rate));
        }

        Iterator<TwoDimensionalMap.Entry<String, String, INDArray>> bslIteratorL = gradients.binaryScoreLayerLeft.iterator();
        while (bslIteratorL.hasNext()) {
            TwoDimensionalMap.Entry<String, String, INDArray> it = bslIteratorL.next();
//            System.err.println(it.getFirstKey() + "    " + it.getSecondKey());
            binaryScoreLayerLeft.get(it.getFirstKey(), it.getSecondKey()).addi(it.getValue().muli(rate));
        }

        // insert the word vector gradients
        for (Map.Entry<String, INDArray> wvEntry : gradients.wordVectors.entrySet()) {
            wordVectors.get(wvEntry.getKey()).addi(wvEntry.getValue().muli(rate));
        }

        // insert the word vector gradients
        for (Map.Entry<String, INDArray> disEntry : gradients.disVectors.entrySet()) {
//            System.err.println(disEntry.getKey());
            disVectors.get(disEntry.getKey()).addi(disEntry.getValue().muli(rate));
        }
    }
}