package bestFirstDepRerank;

import edu.stanford.nlp.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.TanhDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.AdaGrad;

import java.util.IdentityHashMap;

/**
 * Created by zhouh on 15-12-12.
 * <p>
 * A Recursive Neural Net with multi W Matrix according to the children labels
 */
public class DepRCNN {
    private final int n;
    public final DepRCNNModel model;
    static int COMPOSE_NUM = 7;

    /** a value that is saved globally for optimization purpose. This is however a bad
     * programming choice and should be removed in later versions.*/
//    private INDArray nonLinearDerivativeTranspose;

    /**
     * for a binarized DepRCNNTree, W is in Rnx2n and b is in Rn
     */
    public DepRCNN(int n, DepRCNNModel model) {
        this.n = n;
        this.model = model;

    }

    /**
     * get the convolution embedding and score of a given child
     * TODO I will add the dis embedding in the future
     *
     * @param father
     * @param child
     * @param disInfo
     * @return
     */
    public Pair<Double, INDArray> convolution(DepRCNNTree father, DepRCNNTree child,
                                              String disInfo) {

//        INDArray bias = Nd4j.ones(1, 1);
        if(child.getVector() == null)
            System.err.println("vector is null!");

        INDArray disEmb = model.getDisVector(disInfo);
        INDArray disEmbTanh = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", disEmb.dup()));

        INDArray convEmb = Nd4j.vstack(father.headEmb, child.getVector(), father.leftWordTanhEmb,
                father.rightWordTanhEmb, child.leftWordTanhEmb, child.rightWordTanhEmb, disEmbTanh);

        child.concateEmb = convEmb;

        String fPos = father.getLabel();
        String cPos = child.getLabel();
        INDArray binaryTransform = model.getBinaryTransform(fPos, cPos);
        INDArray binaryScoreLayer = model.getBinaryScoreLayer(fPos, cPos);
        convEmb = binaryTransform.mmul(convEmb);
//        try{
//
//        }
//        catch(Exception e){
//            System.err.println("");
//
//        }

        convEmb = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", convEmb));
        double currentNodeScore = binaryScoreLayer.mmul(convEmb).getDouble(0);

        return Pair.makePair(currentNodeScore, convEmb);

    }

    /**
     * return the max pooling result over the convolution embedding
     * #NOTE that the returned pooling argmax vector is a row vector
     *
     * @param convEmb
     * @return
     */
    public Pair<INDArray, INDArray> pooling(INDArray convEmb) {

        if(convEmb.shape()[1] == 1){
            INDArray transpose = convEmb.transpose();
            return Pair.makePair(convEmb, Nd4j.zeros(transpose.shape()));

        }
        return Pair.makePair(Nd4j.max(convEmb, 1), Nd4j.argMax(convEmb, 1));
    }


    /**
     * forward the DepRCNNTree neural net and score each non-leaf nodes
     * in the DepRCNNTree
     *
     * @param t
     * @return
     */
    public double feedForwardAndScore(DepRCNNTree t) {

        int numChildren = t.numChildren();

        if (numChildren == 0) { //leaf vector, we do not score the leaf node

            t.vector = t.headEmb;

            return 0;

        } else {

            double score = 0;
            INDArray[] convArray = new INDArray[numChildren];
            int i = 0;


            // get score from children
            for (DepRCNNTree lChild : t.leftChildren)
                score += feedForwardAndScore(lChild);
            for (DepRCNNTree rChild : t.rightChildren)
                score += feedForwardAndScore(rChild);

            int lChildIndex = 0;
            int leftChildNum = t.leftChildren.size();
            for (DepRCNNTree lChild : t.leftChildren) {

                // get the convolution embedding and score
                String dis = "L/"+ lChildIndex++ +"/"+leftChildNum+"/"+numChildren;
                lChild.disInfo = dis;
                Pair<Double, INDArray> convPair = convolution(t, lChild, dis);
                score += convPair.first;
                lChild.convolutionEmb = convPair.second;
                convArray[i++] = (lChild.convolutionEmb);
            }

            int rChildIndex = 0;
            int rightChildNum = t.leftChildren.size();
            for (DepRCNNTree rChild : t.rightChildren) {

                // get the convolution embedding and score
                String dis = "R/"+ rChildIndex++ +"/"+rightChildNum+"/"+numChildren;
                rChild.disInfo = dis;
                Pair<Double, INDArray> convPair = convolution(t, rChild, dis);
                score += convPair.first;
                rChild.convolutionEmb = convPair.second;
                convArray[i++] = (rChild.convolutionEmb);
            }

            // turn to a matrix by concatenating horizontally.
            INDArray convEmbed = Nd4j.hstack(convArray);

//            System.err.println(convEmbed.shape()[0] +" convEMb " +convEmbed.shape()[1]);

            Pair<INDArray, INDArray> poolingResult = pooling(convEmbed);

//            System.err.println(poolingResult.first.shape()[0] +" poolingResult.first " +poolingResult.first.shape()[1]);
//            System.err.println(poolingResult.second.shape()[0] +" poolingResult.second " +poolingResult.second.shape()[1]);

            // set the composition vector for the father node
            t.vector = poolingResult.first;

//            System.err.println(t.vector.shape()[0] +" " +t.vector.shape()[1]);

//            if(t.vector.shape()[0] == 1)
//                System.err.println("");

            // get the array to record what have been used in the max pooling
            double[][] recordArray = new double[numChildren][getDimension()];
            INDArray maxRecordArray = poolingResult.second;

            for (int j = 0; j < maxRecordArray.size(1); j++)
                recordArray[maxRecordArray.getInt(j)][j] = 1.0;

//                try{
//
//                }
//                catch (Exception e){
//
//                    System.err.println("");
//                }


                // set the record Array for every child of the root
                int childIndex = 0;
            for (DepRCNNTree lChild : t.leftChildren) {
                lChild.poolingRecordArray = Nd4j.create(recordArray[childIndex++], new int[]{getDimension(), 1});
            }
            for (DepRCNNTree rChild : t.rightChildren)
                rChild.poolingRecordArray =  Nd4j.create(recordArray[childIndex++], new int[]{getDimension(), 1});


            return score;

        }
    }

    /**
     * back propagation of a given tree
     *
     * @param t
     * @param error
     * @param gradient
     * @param bpRate
     */
    public void backProp(DepRCNNTree t, INDArray error, DepRCNNModel gradient, double bpRate) {

        int numChildren = t.numChildren();

        // update the leaf node
        if(numChildren == 0){
            gradient.getOrInsertWordVector(t.head).addi(error);
            return ;
        }

        for (DepRCNNTree lChild : t.leftChildren) {

            String fPos = t.getLabel();
            String cPos = lChild.getLabel();
            INDArray binaryTransform = model.getBinaryTransform(fPos, cPos);
            INDArray binaryScoreLayer = model.getBinaryScoreLayer(fPos, cPos);

            // get the derivative from the score layer
            INDArray convEmbTanhDerivative = Nd4j.getExecutioner()
                    .execAndReturn(new TanhDerivative(lChild.convolutionEmb.dup()));
            INDArray deltaFromScore = binaryScoreLayer.transpose().mul(convEmbTanhDerivative);
            if(bpRate != 1.0)
                deltaFromScore.muli(bpRate);

            // get the derivative from the father node
            INDArray deltaFromFather = lChild.poolingRecordArray.mul(error); ///////////////// more check

            // get the total delta
            INDArray delta = deltaFromScore.addi(deltaFromFather);

            // get the error for the composition embedding
            INDArray compositionEmbDelta = binaryTransform.transpose().mmul(delta);
            INDArray[] embGradients = splitInto(compositionEmbDelta);

            // update the score layer
            gradient.getOrInsertBinaryScoreLayer(fPos, cPos).addi(lChild.convolutionEmb.transpose().muli(bpRate));  ////// more check

            // update the tranform matrix
            INDArray transformGradients = delta.mmul(lChild.concateEmb.transpose());
            gradient.getOrInsertBinaryTransform(fPos, cPos).addi(transformGradients);

            // update the concatenate embedding
            INDArray concateEmbTanhDerivative = Nd4j.getExecutioner()
                    .execAndReturn(new TanhDerivative(lChild.concateEmb.dup()));
            INDArray[] embTanhDerivative = splitInto(concateEmbTanhDerivative);
            // head word gradients of father
            gradient.getOrInsertWordVector(t.head).addi(embGradients[0].muli(embTanhDerivative[0]));
            //child phrase embedding gradients
            INDArray gradientForChild = embGradients[1].muli(embTanhDerivative[1]);
            // father left word
            gradient.getOrInsertWordVector(t.leftWord).addi(embGradients[2].muli(embTanhDerivative[2]));
            //father right word
            gradient.getOrInsertWordVector(t.rightWord).addi(embGradients[3].muli(embTanhDerivative[3]));
            // child left word
            gradient.getOrInsertWordVector(lChild.leftWord).addi(embGradients[4].muli(embTanhDerivative[4]));
            // child right word
            gradient.getOrInsertWordVector(lChild.rightWord).addi(embGradients[5].muli(embTanhDerivative[5]));

            gradient.getOrInsertDisVector(lChild.disInfo).addi(embGradients[6].muli(embTanhDerivative[6]));
            backProp(lChild, gradientForChild, gradient, bpRate);
        }

        for (DepRCNNTree rChild : t.rightChildren) {

            String fPos = t.getLabel();
            String cPos = rChild.getLabel();
            INDArray binaryTransform = model.getBinaryTransform(fPos, cPos);
            INDArray binaryScoreLayer = model.getBinaryScoreLayer(fPos, cPos);

            // get the derivative from the score layer
            INDArray convEmbTanhDerivative = Nd4j.getExecutioner()
                    .execAndReturn(new TanhDerivative(rChild.convolutionEmb.dup()));
            INDArray deltaFromScore = binaryScoreLayer.transpose().mul(convEmbTanhDerivative);
            if(bpRate != 1.0)
                deltaFromScore.muli(bpRate);

            // get the derivative from the father node
            INDArray deltaFromFather = rChild.poolingRecordArray.mul(error); ///////////////// more check

            // get the total delta
            INDArray delta = deltaFromScore.addi(deltaFromFather);

            // get the error for the composition embedding
            INDArray compositionEmbDelta = binaryTransform.transpose().mmul(delta);
            INDArray[] embGradients = splitInto(compositionEmbDelta);

            // update the score layer
            gradient.getOrInsertBinaryScoreLayer(fPos, cPos).addi(rChild.convolutionEmb.transpose().muli(bpRate));

            // update the tranform matrix
            INDArray transformGradients = delta.mmul(rChild.concateEmb.transpose());
            gradient.getOrInsertBinaryTransform(fPos, cPos).addi(transformGradients);

            // update the concatenate embedding
            INDArray concateEmbTanhDerivative = Nd4j.getExecutioner()
                    .execAndReturn(new TanhDerivative(rChild.concateEmb.dup()));
            INDArray[] embTanhDerivative = splitInto(concateEmbTanhDerivative);
            // head word gradients of father
            gradient.getOrInsertWordVector(t.head).addi(embGradients[0].muli(embTanhDerivative[0]));
            //child phrase embedding gradients
            INDArray gradientForChild = embGradients[1].muli(embTanhDerivative[1]);
            // father left word
            gradient.getOrInsertWordVector(t.leftWord).addi(embGradients[2].muli(embTanhDerivative[2]));
            //father right word
            gradient.getOrInsertWordVector(t.rightWord).addi(embGradients[3].muli(embTanhDerivative[3]));
            // child left word
            gradient.getOrInsertWordVector(rChild.leftWord).addi(embGradients[4].muli(embTanhDerivative[4]));
            // child right word
            gradient.getOrInsertWordVector(rChild.rightWord).addi(embGradients[5].muli(embTanhDerivative[5]));
            gradient.getOrInsertDisVector(rChild.disInfo).addi(embGradients[6].muli(embTanhDerivative[6]));

            backProp(rChild, gradientForChild, gradient, bpRate);
        }

    }

    public INDArray[] splitInto(INDArray childrenGradients) {

        INDArray[] retval = new INDArray[COMPOSE_NUM];
        int begin = 0;
        for (int i = 0; i < retval.length; i++) {

            retval[i] = childrenGradients.get(NDArrayIndex.interval(begin, begin + n));
            begin += n;
        }

        return retval;
    }


    public int getDimension() {
        return this.n;
    }

    public double getScore(DepRCNNTree DepRCNNTree) {

        return feedForwardAndScore(DepRCNNTree);
    }

    public void updateGradients(DepRCNNModel gradient, IdentityHashMap<INDArray, AdaGrad> gradientSquareMap,
                                int batchSize, double fRegRate) {
        model.updateModel(gradient, gradientSquareMap, batchSize, fRegRate);
    }
}

