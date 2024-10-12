public class NeuralNetwork {

  Layer[] layers;
  int size;

  // initalize the 1st layer of a neural network
  public NeuralNetwork(float[][] inputs, int layer, int size) {
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

}
