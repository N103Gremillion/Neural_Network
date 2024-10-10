class NeuralNetwork {

  int totalLayers;
  Perceptron[][] network;
  float learningRate;

  public NeuralNetwork(int totalLayers, float learningRate, int[] columnSizes){

    this.totalLayers = totalLayers;
    this.learningRate = learningRate;
    // use a jagged array since not all layers will have the same size
    this.network = new Perceptron[totalLayers][];
    // initalize the column size of each row
    for (int i = 0; i < columnSizes.length; i++){
      this.network[i] = new Perceptron[columnSizes[i]];
    }
  }

}