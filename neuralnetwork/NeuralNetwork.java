package only.live.once.memory.neuralnetwork.neuralnetwork;


import android.util.Log;

import only.live.once.memory.neuralnetwork.neuralnetwork.callback.INeuralNetworkCallback;
import only.live.once.memory.neuralnetwork.neuralnetwork.exception.NotSameInputOutputSizeException;
import only.live.once.memory.neuralnetwork.neuralnetwork.exception.ZeroInputDimensionException;
import only.live.once.memory.neuralnetwork.neuralnetwork.exception.ZeroInputElementsException;
import only.live.once.memory.neuralnetwork.neuralnetwork.exception.ZeroNeuronsException;
import only.live.once.memory.neuralnetwork.neuralnetwork.parser.result.BinaryResultParser;
import only.live.once.memory.neuralnetwork.neuralnetwork.parser.result.IResultParser;
import only.live.once.memory.neuralnetwork.neuralnetwork.transfer.ITransferFunction;
import only.live.once.memory.neuralnetwork.neuralnetwork.transfer.SigmoidFunction;
import only.live.once.memory.neuralnetwork.neuralnetwork.utils.Utils;
import only.live.once.memory.neuralnetwork.neuralnetwork.entity.Result;
import only.live.once.memory.neuralnetwork.neuralnetwork.entity.Error;

/**
 * Engine of neural network. This class do all necessary work to learn from input and output data and generate a result
 * @author jlmd
 */
public class NeuralNetwork {
    private int nElements;
    private int dimension;
    private float[][] inputs;
    private int[] outputs;
    private float[] bias;
    private float[] vWeights;
    private float[][] wWeights;
    private float fOut;
    private int neurons;
    private float bOut;
    // Default iterations limit
    private int iterationsLimit = 10000;

    private Analyzer analyzer;
    private Learner learner;
    private IResultParser resultParser;
    private ITransferFunction transferFunction;

    private INeuralNetworkCallback neuralNetworkCallback = null;

    public NeuralNetwork (float[][] inputs, int[] output, INeuralNetworkCallback neuralNetworkCallback) {
        bOut = Utils.randFloat(-0.5f, 0.5f);
        this.neuralNetworkCallback = neuralNetworkCallback;
        // Default transfer function
        this.transferFunction = new SigmoidFunction();
        // Default result parser
        this.resultParser = new BinaryResultParser();

        this.inputs = inputs;
        this.outputs = output;

        this.nElements = output.length;
        try {
            this.dimension = inputs[0].length;
        } catch (ArrayIndexOutOfBoundsException e){
            neuralNetworkCallback.failure(Error.ZERO_INPUT_ELEMENTS);
        }

        // Default num neurons = dimension
        this.neurons = dimension;
    }

    /**
     * Starts a asynchronous thread for do all neural network work
     * Calls a callback on finished
     */
    public void startLearning(){
        try {
            if (inputs.length != outputs.length)
                throw new NotSameInputOutputSizeException();
            if (inputs.length == 0)
                throw new ZeroInputElementsException();

            HiddenLayerNeuron hiddenLayerNeuron = new HiddenLayerNeuron(neurons, dimension);
            bias = hiddenLayerNeuron.getBias();
            vWeights = hiddenLayerNeuron.getVWeights();
            wWeights = hiddenLayerNeuron.getWWeights();

            new NeuralNetworkThread().start();

        } catch (NotSameInputOutputSizeException e) {
            neuralNetworkCallback.failure(Error.NOT_SAME_INPUT_OUTPUT);
        } catch (ZeroInputDimensionException e) {
            neuralNetworkCallback.failure(Error.ZERO_INPUT_DIMENSION);
        } catch (ZeroInputElementsException e) {
            neuralNetworkCallback.failure(Error.ZERO_INPUT_ELEMENTS);
        } catch (ZeroNeuronsException e) {
            neuralNetworkCallback.failure(Error.ZERO_NEURONS);
        }
    }

    /**
     * Get a row of inputs elements
     * @param row Row to obtain
     * @return obtained row
     */
    private float[] getRowElements(int row){
        float[] elements = new float[dimension];
        for (int i = 0; i<dimension; i++){
            elements[i] = this.inputs[row][i];
        }
        return elements;
    }

    public void setTransferFunction(ITransferFunction transferFunction){
        this.transferFunction = transferFunction;
    }

    public int getNeurons() {
        return neurons;
    }

    public void setNeurons(int neurons) {
        this.neurons = neurons;
    }

    public void setResultParser(IResultParser resultParser) {
        this.resultParser = resultParser;
    }

    public int getIterationsLimit() {
        return iterationsLimit;
    }

    public void setIterationsLimit(int iterationsLimit) {
        this.iterationsLimit = iterationsLimit;
    }

    /**
     * Kernel of neural network. This code run on a separate thread and do all necessary work to make
     * the neural network learn.
     */
    public class NeuralNetworkThread extends Thread {
        @Override
        public void run() {
            float quadraticError = 0;
            float[] f;
            int success = 0;
            for (int i = 0; i<iterationsLimit; i++) {
                Log.i("Running " + i,"~");
                success = 0;
                for (int j = 0; j<nElements; j++) {
                    float[] rowElements = getRowElements(j);
                    analyzer = new Analyzer(rowElements, wWeights, bias, vWeights, bOut, neurons, transferFunction, dimension);
                    f = analyzer.getFOutArray();
                    fOut = analyzer.getFOut();
                    learner = new Learner(outputs[j], fOut, f, vWeights, wWeights, bias, bOut, neurons, rowElements, dimension);
                    vWeights = learner.getVWeights();
                    wWeights = learner.getWWeights();
                    bias = learner.getBias();
                    bOut = learner.getBOut();
                    success = resultParser.countSuccesses(success, fOut, outputs[j]);
                    quadraticError += Math.pow(((outputs[j] - fOut)), 2);
                }
                quadraticError *= 0.5f;
            }
            float successPercentage = (success / (float)nElements) * 100;
            Result result = new Result(analyzer, resultParser, successPercentage, quadraticError);
            neuralNetworkCallback.success(result);
        }
    }
}
