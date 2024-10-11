
/*************Class representing a perceptron*******************/
public class Layer {

  // this is the final value the perceptron will hold (Note: this will be the input x into the next layer if not on the output layer)
  Matrix activationValues;
  // this is the weightedSum + bias / which sould be plugged into the sigmoid function to get a valid activation value (since this is 1 perceptron there is 1 z value)
  Matrix weights;
  Matrix biases;
  int layer;

  public Layer(Matrix inputs, Matrix weights, Matrix biases, int layer){
    if (layer == 0){
      this.activationValues = inputs;
    }
    else {
      this.weights = weights;
      this.biases = biases;
      this.activationValues = calculateActivationValues(inputs);
    }
  }

  Matrix calculateActivationValues(Matrix prevActivations){
    
  }

}