package bestFirstDepRerank;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.concurrent.MulticoreWrapper;
import edu.stanford.nlp.util.concurrent.ThreadsafeProcessor;
import nndep.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdaGrad;

import java.io.IOException;
import java.util.*;
import java.util.PriorityQueue;

import static java.lang.System.exit;
import static java.lang.System.setErr;

/**
 * Created by zhouh on 15-12-31.
 */
public class BestFirstDepReranker {

    DependencyParser parser;
    RerankScorer scorer;
    LSTMScorer lstmScorer;

    // map from neural network paras to the gradient square history
    IdentityHashMap<INDArray, AdaGrad> hGradientSquareMap;
    IdentityHashMap<INDArray, AdaGrad> cGradientSquareMap;

    Config config;

    public BestFirstDepReranker(DependencyParser parser, Properties properties) {
        this.parser = parser;
        config = new Config(properties);
    }

    public void setBaseParser(DependencyParser parser) {
        this.parser = parser;
    }

    /**
     * Explicitly specifies the number of arguments expected with
     * particular command line options.
     */
    private static final Map<String, Integer> numArgs = new HashMap<>();

    static {
        numArgs.put("textFile", 1);
        numArgs.put("outFile", 1);
    }

    public static void main(String[] args) {
        Properties rerankerProps = StringUtils.argsToProperties(args, numArgs);
        Properties baseParserProps = StringUtils.argsToProperties(new String[]{}, numArgs);


//        int foldNum = 10;
//
//        for (int i = 1; i <= 10; i++) {
//
//        }

        if (rerankerProps.containsKey("trainFile") || rerankerProps.containsKey("trainDir")) {

            BestFirstDepReranker reranker =
                    new BestFirstDepReranker(null, rerankerProps);
            try {
                reranker.train(
                        rerankerProps.getProperty("trainFile"),
                        rerankerProps.getProperty("trainDir"),
                        rerankerProps.getProperty("devFile"),
                        rerankerProps.getProperty("model"),
                        rerankerProps.getProperty("embedFile"),
                        rerankerProps.getProperty("baseModel"));
            } catch (IOException e) {
                e.printStackTrace();
            }

        } else if (rerankerProps.containsKey("testFile")) {

            DependencyParser baseParser = new DependencyParser(baseParserProps);
            baseParser.loadModelFile(rerankerProps.getProperty("baseModel"));
            BestFirstDepReranker reranker =
                    new BestFirstDepReranker(baseParser, rerankerProps);
            DepRCNNModel hModel = DepRCNNModel.readModel(rerankerProps.getProperty("model") + ".h");
            DepRCNN hRNN = new DepRCNN(reranker.config.nRNNDim, hModel);
            DepRCNNModel cModel = DepRCNNModel.readModel(rerankerProps.getProperty("model") + ".c");
            DepRCNN cRNN = new DepRCNN(reranker.config.nRNNDim, cModel);
            reranker.scorer = new RerankScorer(hRNN, cRNN);

            reranker.test(rerankerProps.getProperty("testFile"), rerankerProps.getProperty("devFile"), rerankerProps.getProperty("output"));

        }

    }

    private List<RerankingExample> genExamples(List<CoreMap> sents, List<DependencyTree> goldTrees) {

        if (sents.size() != goldTrees.size()) {
            System.err.println("Sents Size and Gold Tree Size are not Consistent!");
            exit(0);
        }

        List<RerankingExample> retval = new ArrayList<>();

        for (int i = 0; i < sents.size(); i++) {
            retval.add(new RerankingExample(goldTrees.get(i), sents.get(i)));
        }

        return retval;

    }

    private DepRCNNModel genModel(String trainFile) {
        List<CoreMap> trainSents = new ArrayList<>();
        List<DependencyTree> trainTrees = new ArrayList<>();
        Util.loadConllFile(trainFile, trainSents, trainTrees);

        return generateRNNModel(trainSents);

    }

    private void train(String trainFile, String foldTrainDir, String devFile, String model,
                       String embedFile, String baseModel) throws IOException {

        // read the training and dev files
        System.err.println("###Reading Training and Dev Files.");

        List<CoreMap> devSents = new ArrayList<>();
        List<DependencyTree> devTrees = new ArrayList<>();
        Util.loadConllFile(devFile, devSents, devTrees);

        // load model of the baseline parser
        if (foldTrainDir == null) {
            DependencyParser parser = new DependencyParser();
            parser.loadModelFile(baseModel);
            setBaseParser(parser);
        }

        DependencyParser parserForTest = new DependencyParser();
        parserForTest.loadModelFile(baseModel);

        // construct the reranking scorer
        // set the recursive neural net for reranking
        // add the word embedding and matrix into the model
        DepRCNNModel hModel = genModel(trainFile);
        DepRCNNModel cModel = genModel(trainFile);
        if (embedFile != null) {
            hModel.readPreTrain(embedFile, config.nRNNDim);
            cModel.readPreTrain(embedFile, config.nRNNDim);

        }
        DepRCNN hRNN = new DepRCNN(config.nRNNDim, hModel);
        DepRCNN cRNN = new DepRCNN(config.nRNNDim, cModel);
        scorer = new RerankScorer(hRNN, cRNN);

//        lstmScorer = new LSTMScorer(100, 0.01, 300);
//        lstmScorer.prepareNetwork(config.adaAlpha, config.regParameter);


        // create the whole ada gradients for training
        hGradientSquareMap = scorer.hRNN.model.createAdaGradSquare(config.adaAlpha);
        cGradientSquareMap = scorer.cRNN.model.createAdaGradSquare(config.adaAlpha);
        DepRCNNModel hEmptyGradients = scorer.hRNN.model.createGradients();
        DepRCNNModel cEmptyGradients = scorer.cRNN.model.createGradients();

        double bestDevUAS = Double.NEGATIVE_INFINITY;

        // training iterations
        for (int iter = 0; iter < config.maxIter; iter++) {

            double loss = 0; // total loss of iteration

            // ten folds training
            for (int fold = 1; fold <= 10; fold++) {

                // dev evaluation
                DependencyParser trainParser = parser;
                if (foldTrainDir != null)
                    setBaseParser(parserForTest);
                Pair<List<DependencyTree>, Double> devResult = dev(devSents, devTrees);
                if (bestDevUAS < devResult.second) {
                    bestDevUAS = devResult.second;
                    System.err.println("New Best UAS: " + bestDevUAS);
                    DepRCNNModel.writeModel(model + ".h", scorer.hRNN.model);
                    DepRCNNModel.writeModel(model + ".c", scorer.cRNN.model);
                }
                setBaseParser(trainParser);

                System.err.println("###Reading Training and Dev Files.");
                List<CoreMap> oneFoldTrainSents = new ArrayList<>();
                List<DependencyTree> oneFoldTrainTrees = new ArrayList<>();
                if (foldTrainDir != null) {
                    String fileName = foldTrainDir + fold + "/" + fold + ".fold";
                    System.err.println("Loading Training File : " + fileName);
                    Util.loadConllFile(fileName, oneFoldTrainSents, oneFoldTrainTrees);
                    // set the parser for this fold
                    DependencyParser parser = new DependencyParser();
                    parser.loadModelFile(foldTrainDir + fold + "/hierarchi.hidden300.model.gz");
                    setBaseParser(parser);
                } else
                    Util.loadConllFile(trainFile, oneFoldTrainSents, oneFoldTrainTrees);

                List<RerankingExample> oneFoldTrainExamples = genExamples(oneFoldTrainSents, oneFoldTrainTrees);


//                Collections.shuffle(oneFoldTrainExamples);
                int numBatches = oneFoldTrainExamples.size() / config.batchSize + 1;

                for (int nBatchIndex = 0; nBatchIndex < numBatches; nBatchIndex++) {

                    System.err.print("The " + nBatchIndex + " ");
                    int start = nBatchIndex * config.batchSize;
                    int end = (nBatchIndex + 1) * config.batchSize;
                    if (end + config.batchSize > oneFoldTrainExamples.size()) {
                        end = oneFoldTrainExamples.size();
                    }

//                System.err.println("nBatchIndex: " + nBatchIndex);

                    double l = batchTraining(oneFoldTrainExamples.subList(start, end), hEmptyGradients, cEmptyGradients);

                    loss += l;

                    scorer.hRNN.updateGradients(hEmptyGradients, hGradientSquareMap, config.batchSize, config.regParameter);
                    scorer.cRNN.updateGradients(cEmptyGradients, cGradientSquareMap, config.batchSize, config.regParameter);

                }  // end #for batch
                if (foldTrainDir == null)
                    break;
            }


            System.err.println("Iteration " + iter + "cost : " + loss);
        }


    }

    private Pair<List<DependencyTree>, Double> dev(List<CoreMap> devSents, List<DependencyTree> goldTrees) {

        System.err.println("Test Size : " + devSents.size());

        List<List<DependencyTree>> kbests = new ArrayList<>();
        List<List<Triple<Double, Double, DependencyTree>>> mixRerankData = new ArrayList<>();
        List<DependencyTree> results = new ArrayList<>();
        for (int i = 0; i < devSents.size(); i++) {

            if (i % 5 == 0)
                System.err.println(i);

            List<Triple<Double, Double, DependencyTree>> kbest = new ArrayList<>();
            Pair<Double, DependencyTree> doubleDependencyTreePair = config.bUseBeamSample ?
                    beamSearchDecode(devSents.get(i), null, null, null, kbest) : null;
//                    bestFirstDecode(devSents.get(i), null, null, null, kbest);

            List<DependencyTree> k = new ArrayList<>();
            for (Triple<Double, Double, DependencyTree> tri : kbest) {
                k.add(tri.third);
            }
            kbests.add(k);
            mixRerankData.add(kbest);
            results.add(doubleDependencyTreePair.second);
        }

//        if(config.bMixReranker) {
//            double UAS = mixRerank(mixRerankData, 0.0, 1.0, 0.005, devSents, goldTrees);
//            return Pair.makePair(results, UAS);
//        }

//        else {
        // compute predict UAS
        Map<String, Double> result = parser.system.evaluate(devSents, results, goldTrees);
        double UAS = result.get("UASwoPunc");
        System.err.println("UAS: " + UAS);

        // compute the kbest oracles
        parser.system.evaluateOracle(devSents, kbests, goldTrees);


        return Pair.makePair(results, UAS);
//        }
    }

    private void mixtest(List<CoreMap> devSents, List<CoreMap> testSents,
                         List<DependencyTree> devGoldTrees, List<DependencyTree> testGoldTrees) {

        System.err.println("Test Size : " + devSents.size());

        List<List<DependencyTree>> kbests = new ArrayList<>();
        List<List<Triple<Double, Double, DependencyTree>>> mixRerankData = new ArrayList<>();

        for (int i = 0; i < devSents.size(); i++) {

            if (i % 5 == 0)
                System.err.println(i);

            List<Triple<Double, Double, DependencyTree>> kbest = new ArrayList<>();
            Pair<Double, DependencyTree> doubleDependencyTreePair = config.bUseBeamSample ?
                    beamSearchDecode(devSents.get(i), null, null, null, kbest) : null;

            List<DependencyTree> k = new ArrayList<>();
            for (Triple<Double, Double, DependencyTree> tri : kbest) {
                k.add(tri.third);
            }
            kbests.add(k);
            mixRerankData.add(kbest);
        }

        double bestAlpha = mixRerank(mixRerankData, 0.0, 1.0, 0.005, devSents, devGoldTrees);

        // get the test result
        List<List<Triple<Double, Double, DependencyTree>>> mixRerankData4Test = new ArrayList<>();
        List<DependencyTree> results = new ArrayList<>();

        for (int i = 0; i < testSents.size(); i++) {

            if (i % 5 == 0)
                System.err.println(i);

            List<Triple<Double, Double, DependencyTree>> kbest = new ArrayList<>();
            Pair<Double, DependencyTree> doubleDependencyTreePair = config.bUseBeamSample ?
                    beamSearchDecode(testSents.get(i), null, null, null, kbest) : null;

            List<DependencyTree> k = new ArrayList<>();
            for (Triple<Double, Double, DependencyTree> tri : kbest) {
                k.add(tri.third);
            }
            kbests.add(k);
            mixRerankData4Test.add(kbest);
            results.add(doubleDependencyTreePair.second);
        }

        // begin to mix-test test data

        List<DependencyTree> mixResult = new ArrayList<>();
        for (List<Triple<Double, Double, DependencyTree>> tris : mixRerankData4Test) {  // for every sentence

            double bestTreeScore = Double.NEGATIVE_INFINITY;
            DependencyTree bestTree = null;

            for (Triple<Double, Double, DependencyTree> tri : tris) { // for every triple

                double score = bestAlpha * tri.first + (1 - bestAlpha) * tri.second;

                if (score > bestTreeScore) {
                    bestTree = tri.third;
                    bestTreeScore = score;
                }

            }

            mixResult.add(bestTree);


        }
        Map<String, Double> r = parser.system.evaluate(testSents, mixResult, testGoldTrees);
        double UAS = r.get("UASwoPunc");

        System.err.println("final test UAS: " + UAS);


    }

    private double mixRerank(List<List<Triple<Double, Double, DependencyTree>>> mixRerankData,
                             double b, double e, double step, List<CoreMap> devSents, List<DependencyTree> goldTrees) {

        double bestUAS = Double.NEGATIVE_INFINITY;
        double bestAlpha = 0;

        for (double apha = b; apha <= e; apha += step) {
            List<DependencyTree> result = new ArrayList<>();
            for (List<Triple<Double, Double, DependencyTree>> tris : mixRerankData) {  // for every sentence

                double bestTreeScore = Double.NEGATIVE_INFINITY;
                DependencyTree bestTree = null;

                for (Triple<Double, Double, DependencyTree> tri : tris) { // for every triple

                    double score = apha * tri.first + (1 - apha) * tri.second;

                    if (score > bestTreeScore) {
                        bestTree = tri.third;
                        bestTreeScore = score;
                    }

                }

                result.add(bestTree);


            }
            Map<String, Double> r = parser.system.evaluate(devSents, result, goldTrees);
            double UAS = r.get("UASwoPunc");

            if (UAS > bestUAS) {
                bestUAS = UAS;
                bestAlpha = apha;
            }

        }

        System.err.println("final dev UAS: " + bestUAS);

        return bestAlpha;
    }

    public void test(String testFile, String devFile, String outputFile) {
        List<CoreMap> devSents = new ArrayList<>();
        List<DependencyTree> devTrees = new ArrayList<DependencyTree>();
        Util.loadConllFile(testFile, devSents, devTrees);
        List<RerankingExample> devExamples = genExamples(devSents, devTrees);

        List<CoreMap> testSents = new ArrayList<>();
        List<DependencyTree> testTrees = new ArrayList<DependencyTree>();
        Util.loadConllFile(testFile, testSents, testTrees);
        List<RerankingExample> testExamples = genExamples(testSents, testTrees);

        mixtest(devSents, testSents, devTrees, testTrees);

    }

    private DepRCNNModel generateRNNModel(List<CoreMap> trainSents) {

        DepRCNNModel retval = new DepRCNNModel(config.nRNNDim);

        // process the word in the training data

        // get all the words and pos in the training data
        List<String> knownWords = new ArrayList<>();
        List<String> knownPos = new ArrayList<>();

        for (CoreMap sent : trainSents) {
            List<CoreLabel> tokens = sent.get(CoreAnnotations.TokensAnnotation.class);

            for (CoreLabel token : tokens) {

                // note that all the words in the vocabulary
                // are lower cased
                knownWords.add(token.word().toLowerCase());
                knownPos.add(token.tag());

            }
        }

        // filter all the words and pos
        knownPos.add(Config.ROOT);
        knownWords.add(Config.ROOT.toLowerCase());
        knownWords.add(Config.ROOTLEFT.toLowerCase());
        knownWords.add(Config.ROOTRIGHT.toLowerCase());
        knownWords.add("<LEFT>".toLowerCase());
        knownWords.add("<RIGHT>".toLowerCase());
        knownWords = Util.generateDict(knownWords, config.wordCutOff);
        knownPos = Util.generateDict(knownPos, 1);

        // generate vocabulary
        HashSet<String> vocabulary = new HashSet<>(knownWords);


        // get the known dis info vocabulary
        HashSet<String> disVocabulary = new HashSet<>(knownWords);
        int maxHeadLen = config.nDisNum;
        for (int i = 0; i < maxHeadLen; i++) {
            for (int j = 0; j < maxHeadLen; j++) {
                for (int k = 0; k < 2 * maxHeadLen; k++) {
                    disVocabulary.add("L/" + i + "/" + j + "/" + k);
                }
            }
        }
        for (int i = 0; i < maxHeadLen; i++) {
            for (int j = 0; j < maxHeadLen; j++) {
                for (int k = 0; k < 2 * maxHeadLen; k++) {
                    disVocabulary.add("R/" + i + "/" + j + "/" + k);
                }
            }
        }

        // insert the unk word
        retval.setVocabulary(vocabulary, disVocabulary);

        // generate binary rule for dependency rnn model
        // every rule has two expand, +"*" means it's the head word of the rule
        TwoDimensionalSet<String, String> binaryRules = new TwoDimensionalSet<>();
        for (String pos1 : knownPos)
            for (String pos2 : knownPos) {
                retval.insertBinaryTransform(pos1, pos2, true);
                retval.insertBinaryScoreLayer(pos1, pos2, true);
            }

        // generate all the word vector in the vocabulary
        for (String word : vocabulary)
            retval.insertWordVector(word, true);
        retval.insertUNKWord(Config.UNKNOWN, true);

        // generate dis vectors
        for (String dis : disVocabulary) {
            retval.insertDisVector(dis, true);
        }
        retval.insertUNKDis();


        return retval;

    }


    public double batchTraining(List<RerankingExample> batchExamples, DepRCNNModel hEmptyGradient, DepRCNNModel cEmptyGradient) {

        double loss = 0; // loss for a batch

        MulticoreWrapper<RerankingExample, Triple<Double, DepRCNNModel, DepRCNNModel>> wrapper =
                new MulticoreWrapper<>(config.trainingThreads, new BestFirstTrainProcessor());

        for (RerankingExample example : batchExamples) {
            wrapper.put(example);
        } // end for one batch

        wrapper.join();

        while (wrapper.peek()) {
            Triple<Double, DepRCNNModel, DepRCNNModel> updates = wrapper.poll();

            if (updates.second != null) {
//                emptyGradient.merge(updates.second, -1.0);
                hEmptyGradient.merge(updates.second, 1.0);
                cEmptyGradient.merge(updates.third, 1.0);
                loss += updates.first;
            }
        }


//        System.err.println("batch loss : " + loss);
        return loss;

    }

    class BestFirstTrainProcessor implements ThreadsafeProcessor<RerankingExample,
            Triple<Double, DepRCNNModel, DepRCNNModel>> {
        @Override
        public Triple<Double, DepRCNNModel, DepRCNNModel> process(RerankingExample example) {

            DepRCNNModel hGradient = new DepRCNNModel(config.nRNNDim);
            hGradient.setVocabulary(scorer.hRNN.model.vocabulary, scorer.hRNN.model.disVocabulary);
            DepRCNNModel cGradient = new DepRCNNModel(config.nRNNDim);
            cGradient.setVocabulary(scorer.cRNN.model.vocabulary, scorer.cRNN.model.disVocabulary);

            Pair<Double, DependencyTree> doubleDependencyTreePair = config.bUseBeamSample ?
                    beamSearchDecode(example.sent, example.goldTree, hGradient, cGradient, null) : null;
//                : bestFirstDecode(example.sent, example.goldTree, hGradient, cGradient, null);

            if (doubleDependencyTreePair.first != 0)
                return Triple.makeTriple(doubleDependencyTreePair.first, hGradient, cGradient);
            else
                return Triple.makeTriple(0.0, null, null);

        }

        @Override
        public ThreadsafeProcessor<RerankingExample,
                Triple<Double, DepRCNNModel, DepRCNNModel>> newInstance() {
            // should be threadsafe
            return this;
        }

    }

    /**
     * get the result of a best-first reranking
     *
     * @param sent
     * @param goldTree
     */
    private Pair<Double, DependencyTree> beamSearchDecode(CoreMap sent,
                                                          DependencyTree goldTree,
                                                          DepRCNNModel hGradient,
                                                          DepRCNNModel cGradient,
                                                          List<Triple<Double, Double, DependencyTree>> kbest) {

        double loss = 0;
        Beam beam = new Beam(config.nBeam);
        Beam nextBeam = new Beam(config.nBeam);
        PriorityQueue<RevisedState> revisedItemsFromOneState = new PriorityQueue<>();
        List<Triple<Double, BeamItem, DepRCNNTree>> trainingData4H = new ArrayList<>();
        List<Triple<Double, BeamItem, DepRCNNTree>> trainingData4C = new ArrayList<>();
        int actSize = sent.get(CoreAnnotations.TokensAnnotation.class).size() * 2;
        // priority queue of revised parsing tree

        BeamItem bestBeamItem;  // record the best state searched until now
        Double bestBeamItemScore = Double.NEGATIVE_INFINITY;

        // get the initial result
        HierarchicalDepState initState = parser.partialGreedyParser(null, -1, sent, false);
        initState.bGold = true;

        // prepare for the first beam
        BeamItem initBeamItem = new BeamItem(initState, null, initState.score / actSize);
        beam.insert(initBeamItem);

        // get the gold revised item
        List<ReviseItem> goldRevisedItems = goldTree == null ? null : parser.getRevisedItem(sent, goldTree);

        if (goldTree != null)
            System.err.println("gold revised item size " + goldRevisedItems.size());


        // add the RCNN reranking score to the beam item
        double rerankingScore = initBeamItem.score;
        DepRCNNTree initRerankingTree4h = DepRCNNTree.HierarchyDepState2RerankingTree(sent, initState, scorer.hRNN.model);
        DepRCNNTree initRerankingTree4c = DepRCNNTree.HierarchyDepState2RerankingTree(sent, initState, scorer.cRNN.model);

        double hScore = scorer.getHScore(initRerankingTree4h);
        double cScore = scorer.getCScore(initRerankingTree4c);
        initBeamItem.setScore(rerankingScore + hScore);
        trainingData4H.add(Triple.makeTriple(hScore, initBeamItem, initRerankingTree4h));
        trainingData4C.add(Triple.makeTriple(cScore, initBeamItem, initRerankingTree4c));

        if (kbest != null)
            kbest.add(Triple.makeTriple(rerankingScore, hScore + cScore, initBeamItem.state.c.tree));

        // set the best beam item
        bestBeamItem = initBeamItem;

        bestBeamItemScore = rerankingScore + hScore + cScore;
        if (!config.hcScore)
            bestBeamItemScore = rerankingScore + cScore;

        for (int iter = 0; iter < config.nOracleDepth; iter++) {

            trainingData4H.clear();

            boolean bGoldState = false;

            // revised for a state in the beam
            while (beam.size() != 0) {
                revisedItemsFromOneState.clear();
                BeamItem currentItem = beam.poll();

                HierarchicalDepState currentState = currentItem.state;
                HierarchicalDepState revisedPoint = currentItem.revisedPoint;
                HierarchicalDepState nextState = currentState;
                currentState = currentState.lastState;

                //begin to revise for the currentState
                while (currentState.lastState != revisedPoint) {

                    int[] label = currentState.actTypeLabel;
                    double[] scores = currentState.actTypeDistribution;
                    int bestAct = nextState.actType;

                    for (int j = 0; j < ParsingSystem.nActTypeNum; j++) {

                        if (j == bestAct) continue; // it's the best action in the table, just skip!

                        double margin = scores[bestAct] - scores[j];

                        //if action margin is larger than max margin, or is the best or unvalid
                        //action, just skip
                        if (margin < config.dMargin &&
                                label[j] == 0)
                            revisedItemsFromOneState.add(new RevisedState(new ReviseItem(currentState.index, j, margin), currentState, -1));

                    }

                    nextState = currentState;
                    currentState = currentState.lastState;
                }

                // prune the revisedItemsFromOneState, and run the parser over the revised state
                for (int k = 0; k < config.nMaxReviseActNum; k++) {
                    if (revisedItemsFromOneState.size() == 0)
                        break;

                    RevisedState revisedState = revisedItemsFromOneState.poll();

                    // set the parameters for early updates in training process
                    if (goldTree != null &&
                            currentItem.state.bGold &&
                            revisedState.item.equals((iter + 1) > goldRevisedItems.size() ? null : goldRevisedItems.get(iter))) {

                        bGoldState = true;

                    }

                    HierarchicalDepState state =
                            parser.partialGreedyParser(revisedState.state, revisedState.item.reviseActID, sent, true);
                    if (goldTree != null && bGoldState) {
                        state.bGold = true;
                        bGoldState = false;


                    } else
                        state.bGold = false;

                    BeamItem currentBeamItem = new BeamItem(state, revisedState.state, state.score / actSize);

                    // add the RCNN reranking score to the beam item
                    double currentRerankingScore = currentBeamItem.score;
                    DepRCNNTree currentRerankingTree4H = DepRCNNTree.HierarchyDepState2RerankingTree(sent, state, scorer.hRNN.model);
                    DepRCNNTree currentRerankingTree4C = DepRCNNTree.HierarchyDepState2RerankingTree(sent, state, scorer.cRNN.model);
                    double currentHScore = scorer.getHScore(currentRerankingTree4H);
                    double currentCScore = scorer.getCScore(currentRerankingTree4C);
                    if (config.bUseBeamRNNSearch) {
                        currentBeamItem.setScore(currentRerankingScore + currentHScore);
                        trainingData4H.add(Triple.makeTriple(currentHScore, currentBeamItem, currentRerankingTree4H));
                        trainingData4C.add(Triple.makeTriple(currentCScore, currentBeamItem, currentRerankingTree4C));

                        if (kbest != null)
                            kbest.add(Triple.makeTriple(currentRerankingScore, currentHScore + currentCScore, currentBeamItem.state.c.tree));
                    }

                    nextBeam.insert(currentBeamItem);

                    double totalScore = currentRerankingScore + currentHScore + currentCScore;
                    if (!config.hcScore)
                        totalScore = currentRerankingScore + currentCScore;
                    if (totalScore > bestBeamItemScore) {
                        bestBeamItem = currentBeamItem;
                        bestBeamItemScore = totalScore;

                    }
                }

            }

            if (config.earlyUpdate && goldTree != null) {
                boolean bUpdateEarly = true;

                for (BeamItem item : nextBeam)
                    if (item.state.bGold)
                        bUpdateEarly = false;

                if (bUpdateEarly) {

                    System.err.println("Early Update! Depth: " + iter);
                    break;
                }
            }


            // exchange the content of next beam and current beam
            Beam tempBeam = beam;
            beam = nextBeam;
            tempBeam.clearAll();
            nextBeam = tempBeam;

        }


        if (goldTree != null) { //updating in training

            Pair<Double, DependencyTree> cPair = beamGoldMarginUpdate(sent, goldTree, trainingData4C, cGradient);
            Pair<Double, DependencyTree> hPair = beamRankingUpdate(sent, goldTree, hGradient, trainingData4H);

            return Pair.makePair(cPair.first + hPair.first, null);

        } else {  // for decoding
            return new Pair<>(loss, bestBeamItem.state.c.tree);
        }

    }
//
//    /**
//     * get the result of a best-first reranking
//     *
//     * @param sent
//     * @param goldTree
//     */
//    private Pair<Double, DependencyTree> bestFirstDecode(CoreMap sent,
//                                                         DependencyTree goldTree,
//                                                         DepRCNNModel goldGradients,
//                                                         DepRCNNModel predictGradients,
//                                                         List<DependencyTree> kbestTrees) {
//
//        /*
//     *   Given a parsing state, get top config.nMaxReviseActNum revisedItem from
//	 *   revisedItemsFromOneState queue and insert them into the queue according to the
//	 *   product of margin and average of log probability of a complete parsing state
//	 *
//	 *   In each step, we peek one revised item form the queue and get the revised CFGTree.
//	 *   Then insert the revised CFGTree into the revisedTrees queue.
//	 *
//	 *   In the end, when the revisedTrees queue euqal the config.nMaxN, stop and return
//	 *   these revised CFGTrees.
//	 */
//
//
//        int actSize = sent.get(CoreAnnotations.TokensAnnotation.class).size() * 2;
//        double loss = 0;  // loss for training
//        // priority queue of revised state to be re-parsed from small to large
//        PriorityQueue<RevisedState> queue = new PriorityQueue<RevisedState>();
//        // priority queue of revised state with reranking score (base score  +  rnn score) and reranking trees
//        List<Triple<Double, HierarchicalDepState, DepRCNNTree>> revisedStates = new ArrayList<>();
//        // priority queue of revised items from one complete parsing state
//        PriorityQueue<RevisedState> revisedItemsFromOneState = new PriorityQueue<RevisedState>();
//
//        // get the initial result
//        HierarchicalDepState initState = parser.partialGreedyParser(null, -1, sent, false);
//
//
//        //add the greedy parser result first.
//        boolean firstRevise = true;
//
////        HierarchicalDepState bestScoredState = initState;
////        HierarchicalDepState bestUASState = initState;
////        double bestScore = initState.score;
////        double bestUAS = parser.system.evaluateOneTree(sent, bestUASState.c.tree, goldTree);
//
//        //loop until the revised tree size to nMaxN
//        while (revisedStates.size() < config.nMaxN) {
//
//		/*
//         * first revise from the optimal tree by the greedy baseline parser
//		 */
//            if (firstRevise) {  //revise from the greedy classifier output
//
//                HierarchicalDepState state = initState;
//                HierarchicalDepState nextState = state;
//                double initStateScore = state.score / actSize;
//
//                DepRCNNTree rerankTree = DepRCNNTree.HierarchyDepState2RerankingTree(sent, state, scorer.recrusiveNN.model);
//
//                // bUseBestFirstRNNSearch : add reranking score in search k-best or not!
//                if (config.bUseBestFirstRNNSearch) {
////                    System.err.println("base score: " + initStateScore);
//                    initStateScore += scorer.getScore(rerankTree);
////                    System.err.println("Reranking Score: " + initStateScore );
//                    revisedStates.add(Triple.makeTriple(initStateScore, state, rerankTree));
//                } else {
//                    if (config.bLSTMScorer) {
//                        revisedStates.add(Triple.makeTriple(0.0, state, rerankTree));
//
//                    } else {
//                        revisedStates.add(Triple.makeTriple(scorer.getScore(rerankTree), state, rerankTree));
//                    }
//                }
//
//                state = state.lastState; //from last state of the final state
//                //because the final state do not need devise
//                revisedItemsFromOneState.clear();
//
//                //get the acts in the greedy state
//                while (state.lastState != null) {
//
//                    int[] label = state.actTypeLabel;
//                    double[] scores = state.actTypeDistribution;
//                    int bestAct = nextState.actType;
//
//                    for (int j = 0; j < ParsingSystem.nActTypeNum; j++) {
//
//                        if (j == bestAct) continue; // it's the best action in the table, just skip!
//
//                        double margin = scores[bestAct] - scores[j];
////                        if (margin < config.dMargin)
////                            System.out.println("margin = " + margin);
//
//                        //if action margin is larger than max margin, or is the best or unvalid
//                        //action, just skip
//                        if (margin < config.dMargin &&
//                                label[j] == 0)
//                            revisedItemsFromOneState.add(new RevisedState(new ReviseItem(-1, j, margin), state, initStateScore));
//
//                    }
//
//                    nextState = state;
//                    state = state.lastState;
//                }
//
//                firstRevise = false;
//                //add the best n reviseItem to revisedState queue
//                for (int k = 0; k < config.nMaxReviseActNum; k++) {
//                    if (revisedItemsFromOneState.size() == 0)
//                        break;
//
//                    queue.add(revisedItemsFromOneState.poll());
//                }
//            } else {  //revise from already revised parsing state
//
//                if (queue.size() == 0) //no candidate in the queue
//                    break;
//
//                RevisedState rs = queue.poll();
//                //generate the new state and insert the revised tree to revisedTree Queue
//
//                HierarchicalDepState state = parser.partialGreedyParser(rs.state, rs.item.reviseActID, sent, true);
//                HierarchicalDepState nextState = state;
//                double initStateScore = state.score / actSize;
//
//                DepRCNNTree rerankTree = DepRCNNTree.HierarchyDepState2RerankingTree(sent, state, scorer.recrusiveNN.model);
//
//                // bUseBestFirstRNNSearch : add reranking score in search k-best or not!
//                if (config.bUseBestFirstRNNSearch) {
////                    System.err.println("base score: " + initStateScore);
//                    initStateScore += scorer.getScore(rerankTree);
////                    System.err.println("Reranking Score: " + initStateScore );
//                    revisedStates.add(Triple.makeTriple(initStateScore, state, rerankTree));
//                } else {
//                    if (config.bLSTMScorer) {
//                        revisedStates.add(Triple.makeTriple(0.0, state, rerankTree));
//
//                    } else {
//                        revisedStates.add(Triple.makeTriple(scorer.getScore(rerankTree), state, rerankTree));
//                    }
//                }
//
//                revisedItemsFromOneState.clear();
//
//                state = state.lastState; //from last state to second last state!
//                //because the last state do not need devise
//
//                //get the inherit revision candidate
//                while (state.lastState != rs.state) { //till the previous revised state
//
//                    int[] label = state.actTypeLabel;
//                    double[] scores = state.actTypeDistribution;
//                    int aptAct = nextState.actType;
//
//                    for (int j = 0; j < ParsingSystem.nActTypeNum; j++) {
//                        if (j == aptAct || label[j] < 0)
//                            continue;
//
//                        double margin = scores[aptAct] - scores[j];
//
//                        //if action margin is larger than max margin, skip
//
//                        if (margin < config.dMargin)
//                            revisedItemsFromOneState.add(new RevisedState(new ReviseItem(-1, j, margin), state, initStateScore));
//                    }
//
//                    nextState = state;
//                    state = state.lastState;
//                }
//            }
//
//            //add the best n reviseItem to revisedState queue
//            for (int k = 0; k < config.nMaxReviseActNum; k++) {
//                if (revisedItemsFromOneState.size() == 0)
//                    break;
//
//                queue.add(revisedItemsFromOneState.poll());
//            }
//        }
//
//        // used for compute k-best oracle
//        if (kbestTrees != null) {
//            for (int i = 0; i < revisedStates.size(); i++)
//                kbestTrees.add(revisedStates.get(i).second.c.tree);
//        }
//
//        //updating in training
//        if (goldTree != null) {
//
//            if (config.bGoldMarginUpdate)
//                return goldMarginUpdate(sent, goldTree, revisedStates, goldGradients, predictGradients);
//            else if (config.bUseExpectUASUpdate)
//                return expectedUASUpdate(sent, goldTree, null, predictGradients, revisedStates);
//            else if (config.bSoftUpdate)
//                return softmaxUpdate(sent, goldTree, null, predictGradients, revisedStates);
//            else if (config.bLSTMScorer)
//                return LSTMSoftmaxUpdate(sent, goldTree, predictGradients, revisedStates);
//            else
//                return rankingUpdate(sent, goldTree, goldGradients, predictGradients, revisedStates);
//
//        } else {  // for decoding
//            HierarchicalDepState bestScoredState = null;
//            double bestScore = Double.NEGATIVE_INFINITY;
//
//
//            if (config.bLSTMScorer) {
//                for (int i = 0; i < revisedStates.size(); i++) {
//
//                    Triple<Double, HierarchicalDepState, DepRCNNTree> revisedState = revisedStates.get(i);
//
//                    double rerankingScore = lstmScorer.getScore(revisedState.second, false) + revisedState.second.score / actSize;
//
//                    if (rerankingScore > bestScore) {
//                        bestScore = rerankingScore;
//                        bestScoredState = revisedState.second;
//                    }
//
//                }
//            } else {
//                for (int i = 0; i < revisedStates.size(); i++) {
//
//                    Triple<Double, HierarchicalDepState, DepRCNNTree> revisedState = revisedStates.get(i);
//
//                    double rerankingScore = revisedState.first + revisedState.second.score / actSize;
//
//                    if (rerankingScore > bestScore) {
//                        bestScore = rerankingScore;
//                        bestScoredState = revisedState.second;
//                    }
//
//                }
//
//            }
//
//            return new Pair<>(loss, bestScoredState.c.tree);
//        }
//
//    }
//
//    private Pair<Double, DependencyTree> softmaxUpdate(CoreMap sent,
//                                                       DependencyTree goldTree,
//                                                       DepRCNNModel goldGradients,
//                                                       DepRCNNModel predictGradients,
//                                                       List<Triple<Double, HierarchicalDepState, DepRCNNTree>> revisedStates) {
//        double loss = 0;
//        double bestScore = Double.NEGATIVE_INFINITY;
//        double bestUAS = Double.NEGATIVE_INFINITY;
//        DepRCNNTree goldRerankTree = null;
//        double goldTreeRerankingScore = 0;
//        int goldIndex = -1;
//
//        double[] gradients;
//
//
//        // get the UAS array and the best reranking score
//        for (int i = 0; i < revisedStates.size(); i++) {
//
//            Triple<Double, HierarchicalDepState, DepRCNNTree> revisedState = revisedStates.get(i);
//            DependencyTree tree = revisedState.second.c.tree;
//
//            // this is the gold UAS with punctuation, which is used to find the gold tree
//            double UASWithPunc = parser.system.getUASWithPunc(sent, tree, goldTree);
//
//            if (UASWithPunc > bestUAS) {
//                bestUAS = UASWithPunc;
//
//                if (bestUAS == 100) {
//                    goldRerankTree = revisedState.third;
//                    goldTreeRerankingScore = revisedState.first;
//                    goldIndex = i;
//
//                }
//            }
//
//
//            if (revisedState.first > bestScore)
//                bestScore = revisedState.first;
//
//        }
//
//        if (bestUAS != 100) {
//            HierarchicalDepState goldState = parser.genGoldState(sent, goldTree);
//            goldRerankTree = DepRCNNTree.HierarchyDepState2RerankingTree(sent, goldState, scorer.recrusiveNN.model);
//
//            goldTreeRerankingScore = scorer.getScore(goldRerankTree);
//
//            revisedStates.add(Triple.makeTriple(goldTreeRerankingScore, null, goldRerankTree));
//            goldIndex = revisedStates.size() - 1;
//
//        }
//
//        double sumGradients = 0;
//        gradients = new double[revisedStates.size()];
//        // get the gradients to be updated
//        for (int i = 0; i < revisedStates.size(); i++) {
//
//            Triple<Double, HierarchicalDepState, DepRCNNTree> revisedState = revisedStates.get(i);
//
//            gradients[i] = Math.exp(revisedState.first - bestScore);
//
//            sumGradients += gradients[i];
//        }
//
//        for (int i = 0; i < revisedStates.size(); i++) {
//
//            gradients[i] = gradients[i] / sumGradients - (i == goldIndex ? 1 : 0);
//        }
//
//
//        // do the updates
//        INDArray errorArray = Nd4j.zeros(config.nRNNDim, 1);
//        for (int i = 0; i < revisedStates.size(); i++) {
//
//            scorer.recrusiveNN.backProp(revisedStates.get(i).third, errorArray, predictGradients, gradients[i]);
//
//        }
//
//        loss = revisedStates.get(goldIndex).first / sumGradients;
//        return new Pair<>(loss, null);
//    }
//
//    private Pair<Double, DependencyTree> LSTMSoftmaxUpdate(CoreMap sent,
//                                                           DependencyTree goldTree,
//                                                           DepRCNNModel predictGradients,
//                                                           List<Triple<Double, HierarchicalDepState, DepRCNNTree>> revisedStates) {
//        double loss = 0;
//        double bestScore = Double.NEGATIVE_INFINITY;
//        double bestUAS = Double.NEGATIVE_INFINITY;
//        int bestUASID = -1;
//
//        double[] gradients;
//        double[] rerankingScore = new double[revisedStates.size()];
//
//
//        // get the UAS array and the best reranking score
//        for (int i = 0; i < revisedStates.size(); i++) {
//
//            Triple<Double, HierarchicalDepState, DepRCNNTree> revisedState = revisedStates.get(i);
//            DependencyTree tree = revisedState.second.c.tree;
//
//            // this is the gold UAS with punctuation, which is used to find the gold tree
//            double UASWithPunc = parser.system.getUASWithPunc(sent, tree, goldTree);
//
//            if (UASWithPunc > bestUAS) {
//                bestUAS = UASWithPunc;
//                bestUASID = i;
//            }
//
//            rerankingScore[i] = lstmScorer.getScore(revisedState.second, false);
//
//
//            if (rerankingScore[i] > bestScore)
//                bestScore = rerankingScore[i];
//
//        }
//
//
//        double sumGradients = 0;
//        gradients = new double[revisedStates.size()];
//        // get the gradients to be updated
//        for (int i = 0; i < revisedStates.size(); i++) {
//
//            Triple<Double, HierarchicalDepState, DepRCNNTree> revisedState = revisedStates.get(i);
//
//            gradients[i] = Math.exp(rerankingScore[i] - bestScore);
//
//            sumGradients += gradients[i];
//        }
//
//        for (int i = 0; i < revisedStates.size(); i++) {
//
//            gradients[i] = gradients[i] / sumGradients - (i == bestUASID ? 1 : 0);
//        }
//
//
//        // do the updates
//        for (int i = 0; i < revisedStates.size(); i++) {
//
//            lstmScorer.backProp(revisedStates.get(i).second, gradients[i]);
//
//        }
//
//        loss = revisedStates.get(bestUASID).first / sumGradients;
//        return new Pair<>(loss, null);
//    }
//
//    private Pair<Double, DependencyTree> expectedUASUpdate(CoreMap sent,
//                                                           DependencyTree goldTree,
//                                                           DepRCNNModel goldGradients,
//                                                           DepRCNNModel predictGradients,
//                                                           List<Triple<Double, HierarchicalDepState, DepRCNNTree>> revisedStates) {
//        double loss = 0;
//        double bestScore = Double.NEGATIVE_INFINITY;
//        int actSize = sent.get(CoreAnnotations.TokensAnnotation.class).size() * 2;
//
//        double[] gradients = new double[revisedStates.size()];
//        double[] UASGradients = new double[revisedStates.size()];
//        double[] UASArray = new double[revisedStates.size()];
//
//
//        // get the UAS array and the best reranking score
//        for (int i = 0; i < revisedStates.size(); i++) {
//
//            Triple<Double, HierarchicalDepState, DepRCNNTree> revisedState = revisedStates.get(i);
//            DependencyTree tree = revisedState.second.c.tree;
//
//            Pair<Double, Double> UASPair = parser.system.evaluateOneTree(sent, tree, goldTree);
//            UASArray[i] = UASPair.first;
//
//
//            if (revisedState.first > bestScore)
//                bestScore = revisedState.first;
//
//        }
//
//        double sumGradients = 0;
//        double sumUASGradients = 0;
//        // get the gradients to be updated
//        for (int i = 0; i < revisedStates.size(); i++) {
//
//            Triple<Double, HierarchicalDepState, DepRCNNTree> revisedState = revisedStates.get(i);
//
//            gradients[i] = Math.exp(revisedState.first - bestScore);
//            UASGradients[i] = gradients[i] * UASArray[i];
//
//            sumGradients += gradients[i];
//            sumUASGradients += UASGradients[i];
//        }
//
//        if (sumUASGradients == 0)
//            return new Pair<>(0.0, null);
//
//
//        // do the updates
//        INDArray errorArray = Nd4j.zeros(config.nRNNDim, 1);
//        for (int i = 0; i < revisedStates.size(); i++) {
//
//            double gradient = (UASGradients[i] / sumUASGradients) - (gradients[i] / sumGradients);
//            gradient = -1 * gradient; /////////////////////////////
////            System.err.println(gradient);
//
//            scorer.recrusiveNN.backProp(revisedStates.get(i).third, errorArray, predictGradients, gradient);
//
//        }
//
//        loss = sumUASGradients / sumGradients;
//        return new Pair<>(loss, null);
//    }
//
//    private Pair<Double, DependencyTree> rankingUpdate(CoreMap sent, DependencyTree goldTree, DepRCNNModel goldGradients, DepRCNNModel predictGradients, List<Triple<Double, HierarchicalDepState, DepRCNNTree>> revisedStates) {
//
//        double loss = 0;
//
//        HierarchicalDepState bestScoredState = null;
//        HierarchicalDepState updateState = null;
//        double bestScore = Double.NEGATIVE_INFINITY;
//        double bestUAS = Double.NEGATIVE_INFINITY;
//        double bestUASStateRerankingScore = 0;
//        DepRCNNTree bestScoreStateRerankTree = null;
//        DepRCNNTree updateStateRerankTree = null;
//
//
//        // get the best scored state for updating
//        for (int i = 0; i < revisedStates.size(); i++) {
//
//            Triple<Double, HierarchicalDepState, DepRCNNTree> revisedState = revisedStates.get(i);
//            DependencyTree tree = revisedState.second.c.tree;
//
//            Pair<Double, Double> UASPair = parser.system.evaluateOneTree(sent, tree, goldTree);
//            double UAS = UASPair.first;
//
//            DepRCNNTree rerankTree = revisedState.third;
//            double rerankingScore = revisedState.first;
//
//
//            if (UAS > bestUAS) {
//                bestUAS = UAS;
//                updateState = revisedState.second;
//                bestUASStateRerankingScore = rerankingScore;
//                updateStateRerankTree = rerankTree;
//            }
//
//            if (rerankingScore > bestScore) {
//                bestScore = rerankingScore;
//                bestScoredState = revisedState.second;
//                bestScoreStateRerankTree = rerankTree;
//            }
//        }
//
//
//        if (bestScoredState != updateState) {
//
//            System.err.println("Update!");
//
//            loss += bestScore - bestUASStateRerankingScore;
//
//            INDArray errorArray = Nd4j.zeros(config.nRNNDim, 1);
//
//            scorer.recrusiveNN.backProp(updateStateRerankTree, errorArray, goldGradients, 1);
//            scorer.recrusiveNN.backProp(bestScoreStateRerankTree, errorArray, predictGradients, 1);
//
//        }
//
//        return new Pair<>(loss, bestScoredState.c.tree);
//    }

    private Pair<Double, DependencyTree>
    beamRankingUpdate(CoreMap sent,
                      DependencyTree goldTree,
                      DepRCNNModel predictGradients,
                      List<Triple<Double, BeamItem, DepRCNNTree>> revisedStates) {

        double loss = 0;

        HierarchicalDepState bestScoredState = null;
        HierarchicalDepState updateState = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        double bestUAS = Double.NEGATIVE_INFINITY;
        double bestUASStateRerankingScore = 0;
        DepRCNNTree bestScoreStateRerankTree = null;
        DepRCNNTree updateStateRerankTree = null;


        // get the best scored state for updating
        for (int i = 0; i < revisedStates.size(); i++) {

            Triple<Double, BeamItem, DepRCNNTree> revisedState = revisedStates.get(i);
            DependencyTree tree = revisedState.second.state.c.tree;

            Pair<Double, Double> UASPair = parser.system.evaluateOneTree(sent, tree, goldTree);
            double UAS = UASPair.first;

            DepRCNNTree rerankTree = revisedState.third;
            double rerankingScore = revisedState.first;

            if (UAS > bestUAS) {
                bestUAS = UAS;
                updateState = revisedState.second.state;
                bestUASStateRerankingScore = rerankingScore;
                updateStateRerankTree = rerankTree;
            }

            if (rerankingScore > bestScore) {
                bestScore = rerankingScore;
                bestScoredState = revisedState.second.state;
                bestScoreStateRerankTree = rerankTree;
            }
        }

        if (bestScoredState != updateState) {

            loss += bestScore - bestUASStateRerankingScore;

            INDArray errorArray = Nd4j.zeros(config.nRNNDim, 1);

            scorer.hRNN.backProp(updateStateRerankTree, errorArray, predictGradients, -1);
            scorer.hRNN.backProp(bestScoreStateRerankTree, errorArray, predictGradients, 1);

        }


        return new Pair<>(loss, null);
    }

    public Pair<Double, DependencyTree> beamGoldMarginUpdate(CoreMap sent,
                                                             DependencyTree goldTree,
                                                             List<Triple<Double, BeamItem, DepRCNNTree>> revisedStates,
                                                             DepRCNNModel predictGradients) {

        double loss = 0;
        HierarchicalDepState goldState;
        double bestScore = Double.NEGATIVE_INFINITY;
        double bestScoredStateUAS = 0;
        DepRCNNTree bestScoreStateRerankTree = null;
        DepRCNNTree goldStateRerankTree;
        double goldScore = 0;

        goldState = parser.genGoldState(sent, goldTree);
        goldStateRerankTree = DepRCNNTree.HierarchyDepState2RerankingTree(sent, goldState, scorer.cRNN.model);

        goldScore = scorer.getCScore(goldStateRerankTree);


        // get the best scored state for updating
        for (int i = 0; i < revisedStates.size(); i++) {

            Triple<Double, BeamItem, DepRCNNTree> itemAndTree = revisedStates.get(i);
            DependencyTree tree = itemAndTree.second.state.c.tree;

            Pair<Double, Double> UASPair = parser.system.evaluateOneTree(sent, tree, goldTree);
            double UAS = UASPair.first;
            double margin = UASPair.second;

            DepRCNNTree rerankTree = itemAndTree.third;
            double rerankingScore = itemAndTree.first;


            rerankingScore += margin * config.dMarginRate;

            if (rerankingScore > bestScore) {
                bestScore = rerankingScore;
                bestScoreStateRerankTree = rerankTree;
                bestScoredStateUAS = UAS;
            }
        }


        boolean isDone = bestScoredStateUAS == 1.0 || (goldScore > bestScore);

        if (!isDone) {

            System.err.println("Update!");

            loss += bestScore - goldScore;

            INDArray errorArray = Nd4j.zeros(config.nRNNDim, 1);

            scorer.cRNN.backProp(goldStateRerankTree, errorArray, predictGradients, -1);
            scorer.cRNN.backProp(bestScoreStateRerankTree, errorArray, predictGradients, 1);

        }

        return new Pair<>(loss, null);

    }
//
//    public Pair<Double, DependencyTree> beamGoldUpdate(CoreMap sent,
//                                                       DependencyTree goldTree,
//                                                       List<Triple<Double, BeamItem, DepRCNNTree>> revisedStates,
//                                                       DepRCNNModel predictGradients) {
//
//        double loss = 0;
//        HierarchicalDepState goldState;
//        double bestScore = Double.NEGATIVE_INFINITY;
//        double bestScoredStateUAS = 0;
//        DepRCNNTree bestScoreStateRerankTree = null;
//        DepRCNNTree goldStateRerankTree;
//        double goldScore = 0;
//
//        goldState = parser.genGoldState(sent, goldTree);
//        goldStateRerankTree = DepRCNNTree.HierarchyDepState2RerankingTree(sent, goldState, scorer.recrusiveNN.model);
//
//        goldScore = scorer.getScore(goldStateRerankTree);
//
//
//        // get the best scored state for updating
//        for (int i = 0; i < revisedStates.size(); i++) {
//
//            Triple<Double, BeamItem, DepRCNNTree> itemAndTree = revisedStates.get(i);
//            DependencyTree tree = itemAndTree.second.state.c.tree;
//
//            Pair<Double, Double> UASPair = parser.system.evaluateOneTree(sent, tree, goldTree);
//            double UAS = UASPair.first;
//
//            DepRCNNTree rerankTree = itemAndTree.third;
//            double rerankingScore = itemAndTree.first;
//
//
//            if (rerankingScore > bestScore) {
//                bestScore = rerankingScore;
//                bestScoreStateRerankTree = rerankTree;
//                bestScoredStateUAS = UAS;
//            }
//        }
//
//
//        boolean isDone = bestScoredStateUAS == 1.0 || (goldScore > bestScore);
//
//        if (!isDone) {
//
//            System.err.println("Update!");
//
//            loss += bestScore - goldScore;
//
//            INDArray errorArray = Nd4j.zeros(config.nRNNDim, 1);
//
//            scorer.recrusiveNN.backProp(goldStateRerankTree, errorArray, predictGradients, -1);
//            scorer.recrusiveNN.backProp(bestScoreStateRerankTree, errorArray, predictGradients, 1);
//
//        }
//
//        return new Pair<>(loss, null);
//
//    }
//
//    /**
//     * gold margin updating for reranking
//     *
//     * @param sent
//     * @param goldTree
//     * @param revisedStates
//     * @param goldGradients
//     * @param predictGradients
//     * @return
//     */
//    public Pair<Double, DependencyTree> goldMarginUpdate(CoreMap sent,
//                                                         DependencyTree goldTree,
//                                                         List<Triple<Double, HierarchicalDepState, DepRCNNTree>> revisedStates,
//                                                         DepRCNNModel goldGradients,
//                                                         DepRCNNModel predictGradients) {
//
//        double loss = 0;
//        HierarchicalDepState bestScoredState = null;
//        HierarchicalDepState goldState;
//        double bestScore = Double.NEGATIVE_INFINITY;
//        double bestScoredStateUAS = 0;
//        DepRCNNTree bestScoreStateRerankTree = null;
//        DepRCNNTree goldStateRerankTree;
//        double goldScore = 0;
//
//        goldState = parser.genGoldState(sent, goldTree);
//        goldStateRerankTree = DepRCNNTree.HierarchyDepState2RerankingTree(sent, goldState, scorer.recrusiveNN.model);
//
//        goldScore = scorer.getScore(goldStateRerankTree);
//
//
//        // get the best scored state for updating
//        for (int i = 0; i < revisedStates.size(); i++) {
//
//            Triple<Double, HierarchicalDepState, DepRCNNTree> revisedState = revisedStates.get(i);
//            DependencyTree tree = revisedState.second.c.tree;
//
//            Pair<Double, Double> UASPair = parser.system.evaluateOneTree(sent, tree, goldTree);
//            double UAS = UASPair.first;
//            double margin = UASPair.second;
//
//            DepRCNNTree rerankTree = revisedState.third;
//            double rerankingScore = revisedState.first;
//
//
//            rerankingScore += margin * config.dMarginRate;
//
//            if (rerankingScore > bestScore) {
//                bestScore = rerankingScore;
//                bestScoredState = revisedState.second;
//                bestScoreStateRerankTree = rerankTree;
//                bestScoredStateUAS = UAS;
//            }
//        }
//
//
//        boolean isDone = bestScoredStateUAS == 1.0 || (goldScore > bestScore);
//
//        if (!isDone) {
//
//            System.err.println("Update!");
//
//            loss += bestScore - goldScore;
//
//            INDArray errorArray = Nd4j.zeros(config.nRNNDim, 1);
//
//            scorer.recrusiveNN.backProp(goldStateRerankTree, errorArray, predictGradients, -1);
//            scorer.recrusiveNN.backProp(bestScoreStateRerankTree, errorArray, predictGradients, 1);
//
//        }
//
//        return new Pair<>(loss, bestScoredState.c.tree);
//
//    }
}

