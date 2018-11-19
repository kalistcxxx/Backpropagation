package only.live.once.memory.neuralnetwork.neuralnetwork.transfer;

/**
 *
 */
public class SoftMaxFunction implements ITransferFunction {
    @Override
    public float transfer(float value) {
        return 0;
    }

    public float[] transfer(float[] input){
        float[] result = new float[input.length];
        float totalSum = 0;
        for(int i = 0; i < input.length; i++){
            totalSum += Math.exp(input[i]);
        }
        for(int i =0; i < input.length; i++){
            result[i] = (float) Math.exp(input[i]) / totalSum;
        }
        return result;
    }
}
