package bestFirstDepRerank;

import nndep.HierarchicalDepState;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by zhouh on 16-1-28.
 *
 */
public class LSTMScorer {
    MultiLayerNetwork lstm;
    INDArray scoreLayer;
    Distribution distribution;
    int lstmLayerSize;
    int hiddenSize;
    INDArray lstmOutput;
    int timeSize;


    public LSTMScorer(int lstmLayerSize, double intiRange, int hiddenSize) {
        this.lstmLayerSize = lstmLayerSize;
        distribution = new org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution(intiRange * -1, intiRange);
        this.hiddenSize = hiddenSize;
    }

    public void prepareNetwork(double lRate, double regRate) {

        //Set up network configuration:
        MultiLayerConfiguration confLSTM = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(lRate)
                .seed(12345)
                .regularization(true)
                .l2(regRate)
                .list(1)
                .layer(0, new GravesLSTM.Builder().nIn(hiddenSize).nOut(lstmLayerSize)
                        .updater(Updater.ADAGRAD)
                        .activation("tanh").weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.01, 0.01)).build())
                .pretrain(false).backprop(true)
                .build();


        lstm = new MultiLayerNetwork(confLSTM);
        lstm.init();
        lstm.setListeners(new ScoreIterationListener(1));

        scoreLayer = Nd4j.rand(new int[]{lstmLayerSize, 1}, distribution);

    }

    /**
     * get the score of the
     *
     * @param state
     * @return
     */
    public double getScore(HierarchicalDepState state, boolean bTrain) {

        INDArray[] hiddens = getInput(state);

        INDArray input = twoDim2ThreeDim(hiddens);

        List<INDArray> layers = lstm.feedForward(input, bTrain);
        INDArray layer = layers.get(layers.size() - 1); // totally 1 layer!

        lstmOutput = layer.tensorAlongDimension(0, 1, 2);;

        return lstmOutput.mmul(scoreLayer).sum(0).getDouble(0);

    }

    private INDArray[] getInput(HierarchicalDepState state) {
        List<INDArray> inputs = new ArrayList<>();

        state = state.lastState;

        while (state.lastState != null) {
            INDArray a = Nd4j.create(state.hiddenLayer.hidden3, new int[]{hiddenSize, 1});
            inputs.add(a);
            state = state.lastState;
        }

        Collections.reverse(inputs);
        int size = inputs.size();
        timeSize = size;
        INDArray[] retval = inputs.toArray(new INDArray[size]);

        return retval;


    }

    public void backProp(HierarchicalDepState state, double error) {

        getScore(state, true);

        INDArray scoreLayerGradient = lstmOutput.mul(error);
        scoreLayerGradient = scoreLayerGradient.sum(0).transpose();
        scoreLayer.subi(scoreLayerGradient);

        INDArray lstmOneTimeGradients = scoreLayer.mul(error).transpose();


        INDArray lstmGradients = lstmOutput.subColumnVector(lstmOneTimeGradients).transpose().reshape(1, lstmLayerSize, timeSize);// the lstmOneTimeGradients need to be a column vector!

        lstm.backpropGradient(lstmGradients);

    }

    /**
     * the input is the row vector arrays!
     *
     * @param input
     * @return
     */
    private INDArray twoDim2ThreeDim(INDArray[] input) {

        INDArray hstack = Nd4j.hstack(input);
        return hstack.reshape(1, hiddenSize, input.length);
    }
}
