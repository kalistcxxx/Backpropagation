class NeuralNetwork{
  float[] input;
  float[] bias;
  float[] hiddenLayout1;
  float[] hiddenLayout2;
  float[][] weight1;
  float[][] weight2;
  float[] output;
  float[] target;
  float summingError = 0;
  float learningRate = 0.5;
  
  float[][] nHiddenLayout;
  float[][][] weight;
  
  
  NeuralNetwork(float[] in, float[] targetOutput){
    this.input = in;
    this.target = targetOutput;
    this.weight1 = new float[2][2];
    this.weight2 = new float[2][2];
    this.bias = new float[2];
    this.output = new float[2];
    setWeight1();
    setWeight2();
    //randomizeWeight(weight1);
    //randomizeWeight(weight2);
    //randomize(bias1);
    //randomize(bias2);
  }
  
  /**
    contructor with multi hidden layer
  **/
  NeuralNetwork(float[] in, float[] targetOutput, int numberOfHidden, int numberEachHiddenLayer){
    this.input = in;
    this.target = targetOutput;
    this.nHiddenLayout = new float[numberOfHidden][numberEachHiddenLayer];
    this.weight = new float[numberOfHidden + 1][numberEachHiddenLayer][in.length];
    this.bias = new float[numberOfHidden + 1];
    this.output = new float[targetOutput.length];
  }
  
  
  void forwardPassing(boolean isRunning){
    for(int i=0;i<nHiddenLayout.length + 1;i++){ //<>// //<>//
      if(i == 0){
        nHiddenLayout[i] = countingValueLayer(nHiddenLayout[i].length, this.input, this.weight[i], bias[i]);
      } else {
        nHiddenLayout[i] = countingValueLayer(nHiddenLayout[i].length, nHiddenLayout[i - 1], this.weight[i], bias[i]);
      }
      if(i == nHiddenLayout.length - 1){ //<>//
        this.output = countingValueLayer(2, nHiddenLayout[i], this.weight[i + 1], bias[i + 1]);
        break;
      }
    }
    summingError();
    println("Summing Error: " + summingError + " ~");
    //showOutput();
  }
  
  
  void backwardPassing(boolean isRunning){
    for(int i = 0; i < weight.length; i++){
      if(i == weight.length - 1){
        for(int ii =0; ii < weight[i].length; ii++){
          for(int j=0; j < weight[i][ii].length; j++){
            weight[i][ii][j] = weight[i][ii][j] - learningRate * (output[ii] - target[ii]) * sigmoid(output[ii], true) * nHiddenLayout[i - 1][ii];
            println(weight[i][ii][j] + " ~");
          }
        }
      } else {
        for(int ii =0; ii< weight[i].length;ii++ ){
          float tErrorAW = 0;
          for(int j=0; j< weight[i][ii].length;j++){
            tErrorAW += (output[j] - target[j]) * sigmoid(output[j], true) * weight[i + 1][j][ii];
          }
          tErrorAW = input[ii] * (sigmoid(nHiddenLayout[i][ii], true)) * tErrorAW;
          for(int j=0; j< weight[i][ii].length;j++){
            weight[i][ii][j] = weight[i][ii][j] - learningRate * tErrorAW;
            println(weight[i][ii][j] + " ~");
          }
        }
      }
    }
  }
  
  float[] getOutput(){
    return this.output;
  }
  
  
  
  
  void forwardPassing(){
    hiddenLayout1 = countingValueLayer(2, this.input, this.weight1, bias[0]);
    this.output = countingValueLayer(2, hiddenLayout1, this.weight2, bias[1]);
    
    // counting Total Error
    summingError();
    
    // show output
    showOutput();
  }
  
  void backwardPassing(){
    updateWeight();
  }
  
  
  /**
    Update backward weight in nn
  **/
  float updateWeight(){
    /**
      update first weight
    **/
    for(int i=0;i<weight1.length;i++){
      float tErrorAW = 0;
      for(int j=0; j< weight1.length;j++){
        tErrorAW += (output[j] - target[j]) * sigmoid(output[j], true) * weight2[j][i];
      }
      tErrorAW = input[i] * (sigmoid(hiddenLayout1[i], true)) * (tErrorAW);
      for(int j=0; j< weight1.length;j++){
        weight1[i][j] = weight1[i][j] - learningRate * tErrorAW;
        println(weight1[i][j] + " ~");
      }
      
    }
    /**
      update second weight
    **/
    for(int i=0;i<weight2.length;i++){
      for(int j=0; j< weight2.length;j++){
        weight2[i][j] = weight2[i][j] - learningRate * (output[i] - target[i]) * sigmoid(output[i], true) * hiddenLayout1[i];
        //println(weight2[i][j] + " ~");
      }
    }
    return 0;
  }
  
  /**
    calculating the Error Total
  **/
  float summingError(){
    for(int i=0;i< output.length;i++){
      this.summingError += 0.5 * (target[i] - output[i]) * (target[i] - output[i]);
    }
    return this.summingError;
  }
  
  /**
    calculating each node of hidden layer.
  **/
  float[] countingValueLayer(int size, float[] inputLayout, float[][] weightLayer, float bias){
    float[] outputLayout = new float[size];
    for(int i=0;i<outputLayout.length;i++){
      float result = 0;
      for(int ii=0;ii<weightLayer[i].length;ii++){
        result += inputLayout[ii] * weightLayer[i][ii];
      }
      outputLayout[i] = sigmoid(result + bias, false);
    }
    return outputLayout;
  }
   
  /**
    
  **/
  void showOutput(){
    for(int i=0;i<output.length;i++){
      println(output[i] + " ");
    }
    println(summingError + " ~");
  }
  
  void randomize(float[] m){
    for(int i=0;i<m.length;i++){
      m[i] = random(-1, 1);
    }
  }
  
  //Dummy data for weight 1
  void setWeight1(){
    weight1[0][0] = 0.15;
    weight1[0][1] = 0.20;
    weight1[1][0] = 0.25;
    weight1[1][1] = 0.30;
    bias[0] = 0.35;
  }
  
  //Dummy data for weight 2
  void setWeight2(){
    weight2[0][0] = 0.40;
    weight2[0][1] = 0.45;
    weight2[1][0] = 0.50;
    weight2[1][1] = 0.55;
    bias[1] = 0.60;
  }
  
  //Dummy data for n hidden layer
  void dummyData(){
    weight[0][0][0] = 0.15;
    weight[0][0][1] = 0.20;
    weight[0][1][0] = 0.25;
    weight[0][1][1] = 0.30;
    weight[1][0][0] = 0.40;
    weight[1][0][1] = 0.45;
    weight[1][1][0] = 0.50;
    weight[1][1][1] = 0.55;
    bias[0] = 0.35;
    bias[1] = 0.60;
  }
  
  
  void randomizeWeight(float[][] m){
    for(int i=0;i<m.length;i++){
      for(int ii = 0; ii<m.length;ii++){
          m[i][ii] = random(-1, 1);
      }
    }
  }
  
  /**---------------------------------------------------------------------**/
  // Activation Function
  float sigmoid(float x, boolean isBackward){
    if(isBackward) return x * (1 - x);
    return (float)(1 / (1 + Math.exp(-x)));
  }
  
  void tanh(float x, boolean isBackward){
    
  }
  
  float RELu(float x, boolean isBackward){
    return Math.max(0, x);
  }
}
