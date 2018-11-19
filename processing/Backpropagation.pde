
float[] input = new float[] {0.05, 0.10};
float[] target = new float[] {0.01, 0.99};
//NeuralNetwork nn = new NeuralNetwork(input, target);
NeuralNetwork nn = new NeuralNetwork(input, target, 1, 2);
void setup(){
    nn.dummyData();
    for(int i = 0;i<15; i++){
      println("Traning: " + i);
      nn.forwardPassing(true);
      nn.backwardPassing(true);
    }
}
void draw(){

}
