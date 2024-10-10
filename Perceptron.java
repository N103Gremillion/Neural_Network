
/*************Class representing a perceptron*******************/
public class Perceptron {

  // this is the final value the perceptron will hold (Note: this will be the input x into the next layer if not on the output layer)
  float activationValue;
  // this is the weightedSum + bias / which sould be plugged into the sigmoid function to get a valid activation value (since this is 1 perceptron there is 1 z value)
  float zValue;
  float bias;
  Matrix inputWeights;
  Matrix inputValues;

  public Perceptron(float[][] xValues, float[][] weights, float bias){

    if (xValues[0].length != weights.length) {
      throw new IllegalArgumentException("Incompatible dimensions for xValues and weights.");
    }

    // create matrix for xvalues and weights
    Matrix xMatrix = new Matrix(xValues.length, xValues[0].length);
    Matrix weightMatrix = new Matrix(weights.length, weights[0].length);

    xMatrix.fillMatrix(xValues);
    weightMatrix.fillMatrix(weights);

    this.inputValues = xMatrix;
    this.inputWeights = weightMatrix;
    this.bias = bias;
    this.zValue = findZvalue();
    this.activationValue = findActivationValue();

  }

  float findZvalue(){
    // this should always be a 1x1 matrix sice this is a single perceptron
    Matrix weightedSumMatrix = this.inputWeights.dotProductMatrices(this.inputValues);
    float weightedSum = weightedSumMatrix.grid[0][0];

    return (weightedSum + this.bias);
  }

  float findActivationValue(){

    // calculate activation using sigmoid function
    double activation = 1 / (1 + (Math.pow(Math.E, -this.zValue)));

    return (float) activation;
  }

}