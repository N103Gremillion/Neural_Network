public class NeuralNetwork {

  Layer[] layers;
  int size;
  int learningRate;

  // initalize the 1st layer of a neural network
  public NeuralNetwork(double[][] inputs, int layer, int size) {
    if (size == 0){
      return;
    }
    this.size = size;
    this.layers = new Layer[size];
    layers[0] = new Layer(inputs, layer);
  }

  // setup the next layer of the network 
  public void forwardPass (Matrix inputs, Matrix weights, Matrix biases, int layer) {
    layers[layer] = new Layer(inputs, weights, biases, layer);
  }

  // formula for the final layer 
  public void backwardPropogate(Matrix expectedOutputs, int layer) {
    // get the error 
    if (layer == this.size){
      layers[layer - 1].calculateBiasGradientForFinalLayer(expectedOutputs);
    }
    else{
      layers[layer - 1].calculateBiasGradient(layers[layer].weights, layers[layer].biasGradient);
    }
    layers[layer - 1].calculateWeightGradient(layers[layer - 2].activationValues);
  }

}
