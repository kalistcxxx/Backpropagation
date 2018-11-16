package only.live.once.memory.neuralnetwork.neuralnetwork.callback;


import only.live.once.memory.neuralnetwork.neuralnetwork.entity.Result;
import only.live.once.memory.neuralnetwork.neuralnetwork.entity.Error;

/**
 * Callback for neural network
 * @author jlmd
 */
public interface INeuralNetworkCallback {
    /**
     * This method is called when neural network finish his work and all is good
     * @param result Entity to save obtained values
     */
    void success(Result result);

    /**
     * This method is called when neural network finish his work and something is not good
     * @param error Entity to save obtained error
     */
    void failure(Error error);
}
