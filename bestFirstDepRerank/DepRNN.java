package bestFirstDepRerank;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.TanhDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdaGrad;

import java.util.IdentityHashMap;

import static java.lang.System.exit;

/**
 * Created by zhouh on 15-12-12.
 * <p>
 * A Recursive Neural Net with multi W Matrix according to the children labels
 */
public class DepRNN {
    private final int n;
    public final DepRNNModel model;

    /** a value that is saved globally for optimization purpose. This is however a bad
     * programming choice and should be removed in later versions.*/
//    private INDArray nonLinearDerivativeTranspose;

    /**
     * for a binarized DepRerankingTree, W is in Rnx2n and b is in Rn
     */
    public DepRNN(int n, DepRNNModel model) {
        this.n = n;
        this.model = model;

    }


    /**
     * compute the vectors of the given DepRerankingTree
     * Assumes that the children vector of this DepRerankingTree have been computed
     *
     * @param currentDepRerankingTree
     * @return
     */
    public INDArray preOutput(DepRerankingTree currentDepRerankingTree) {

        INDArray preOutput;
        int numChildren = currentDepRerankingTree.numChildren();

        /*
         * compute W * (a;b) for a given DepRerankingTree
         */
        if (numChildren == 0)
            preOutput = model.getWordVector(currentDepRerankingTree);
        else {  // numChild == 2
            //perform composition
            INDArray bias = Nd4j.ones(1, 1);
            INDArray concat = Nd4j.vstack(
                    currentDepRerankingTree.getChild(0).getVector(),
                    currentDepRerankingTree.getChild(1).getVector(),
                    bias);
            preOutput = this.model.getBinaryTransform(currentDepRerankingTree).mmul(concat);

        }
        return preOutput;
    }


    /**
     * feedforward the DepRerankingTree neural net and score each non-leaf nodes
     * in the DepRerankingTree
     *
     * @param t
     * @return
     */
    public double feedForwardAndScore(DepRerankingTree t) {

        if (t.numChildren() == 0) { //leaf vector, we do not score the leaf node

            INDArray preOutput = this.preOutput(t);
            t.setPreOutput(preOutput);
            INDArray nonLinear = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", preOutput.dup()));
            t.setVector(nonLinear);
            return 0;

        } else { //binary

            //do these recursive calls in parallel in future
            double leftChildrenScoreSum = this.feedForwardAndScore(t.getChild(0));
            double rightChildrenScoreSum = this.feedForwardAndScore(t.getChild(1));

            //perform user defined composition
            INDArray preOutput = this.preOutput(t);
            t.setPreOutput(preOutput);
            INDArray nonLinear = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", preOutput.dup()));
            t.setVector(nonLinear);

            double currentNodeScore = model.getBinaryScoreLayer(t).mmul(nonLinear).getDouble(0);

            return currentNodeScore + leftChildrenScoreSum + rightChildrenScoreSum;
        }
    }

    public void backProp(DepRerankingTree t, INDArray error, DepRNNModel gradient) {
        //error term gives us del loss/ del y (y is the output of this node)

        if (t.numChildren() == 0) {
            //fine tune leaf vector embeddings
            gradient.getOrInsertWordVector(t).addi(error);
            return;
        } else {  // children num  ==  1 or 2

            boolean bBinary = t.numChildren() == 2;

            INDArray currentDepRerankingTreeNodeNonlinearVector = t.getVector();
            //nonLinear Derivative = [g'(Wx+b)]
            //Be careful that execAndReturn works on the same copy so duplicate the INDArray
            INDArray currentDepRerankingTreeNodeNonlinearVectorDerivative = Nd4j.getExecutioner()
                    .execAndReturn(new TanhDerivative(currentDepRerankingTreeNodeNonlinearVector.dup()));

            /*
             * compute the gradients according to current node score
             */
            INDArray scoreLayer = model.getBinaryScoreLayer(t);
            INDArray transformLayer = model.getBinaryTransform(t);

            INDArray currentScoreGradients = scoreLayer.transpose().mul(currentDepRerankingTreeNodeNonlinearVectorDerivative);


            /*
             * get the total gradients of current linear layer of nets
             * the total gradients include gradient from parent : error
             *                         and from the score layer : currentScoreGradients
             */
            INDArray currentTotalGradients = currentScoreGradients.add(error);

            /*
             * get the gradients for children
             */
            INDArray childrenGradients = transformLayer.transpose().mmul(currentTotalGradients);


            /*
             * update the gradients of current DepRerankingTree node
             */

            gradient.getOrInsertBinaryScoreLayer(t).addi(currentDepRerankingTreeNodeNonlinearVector.transpose());

                /*
                 * get the child concatenate vector
                 */
            INDArray bias = Nd4j.ones(1, 1);
            INDArray concat = Nd4j.vstack(
                    t.getChild(0).getVector(),
                    t.getChild(1).getVector(),
                    bias);
                /*
                 * get and update the transform gradients
                 */
            INDArray transformGradients = currentTotalGradients.mmul(concat.transpose());
            gradient.getOrInsertBinaryTransform(t).addi(transformGradients);

                /*
                 * back-propagate the gradients to children
                 */
            INDArray leftGradients = leftDerivative(childrenGradients);
            INDArray rightGradients = rightDerivative(childrenGradients);
            INDArray leftNonlinearDerivative = Nd4j.getExecutioner()
                    .execAndReturn(new TanhDerivative(t.getChild(0).getVector().dup()));
            INDArray rightNonlinearDerivative = Nd4j.getExecutioner()
                    .execAndReturn(new TanhDerivative(t.getChild(1).getVector().dup()));

            INDArray leftError = leftGradients.mul(leftNonlinearDerivative);
            INDArray rightError = rightGradients.mul(rightNonlinearDerivative);

            // left child
            backProp(t.getChild(0), leftError, gradient);

            // right child
            backProp(t.getChild(1), rightError, gradient);

        }
    }

    public INDArray leftDerivative(INDArray childrenGradients) {

        INDArray leftDerivative = Nd4j.zeros(this.n, 1);
        for (int row = 0; row < this.n; row++) {
            leftDerivative.putRow(row, childrenGradients.getRow(row));
        }

        return leftDerivative;
    }

    public INDArray rightDerivative(INDArray childrenGradients) {

        INDArray rightDerivative = Nd4j.zeros(this.n, 1);
        for (int row = this.n; row < 2 * this.n; row++) {
            rightDerivative.putRow(row - this.n, childrenGradients.getRow(row));
        }

        return rightDerivative;
    }

    public int getDimension() {
        return this.n;
    }

    public double getScore(DepRerankingTree DepRerankingTree) {

        return feedForwardAndScore(DepRerankingTree);
    }

    public void updateGradients(DepRNNModel gradient, IdentityHashMap<INDArray, AdaGrad> gradientSquareMap,
                                int batchSize, double fRegRate) {
        model.updateModel(gradient, gradientSquareMap, batchSize, fRegRate);
    }
}

