/*************Class representing a perceptron*******************/
public class Layer {

  Matrix activationValues;
  Matrix zValues;
  Matrix weights;
  Matrix biases;
  Matrix biasGradient;
  Matrix weightGradient;
  int layer;

  // only for the opening layer
  public Layer(double[][] inputs, int layer) {
      this.activationValues = new Matrix(inputs);
      this.layer = layer;
  }

  // for every layer but layer 0
  public Layer(Matrix inputs, Matrix weights, Matrix biases, int layer) {
      this.weights = weights;
      this.biases = biases;
      this.zValues = calculateZvalues(inputs, weights, biases);
      this.activationValues = sigmoidFunction(zValues);
      this.layer = layer;
  }

  private static Matrix calculateZvalues(Matrix prevActivations, Matrix weights, Matrix biases) {
      Matrix dottedMatrixWithoutBias = weights.dotProductMatrices(prevActivations);
      Matrix result = dottedMatrixWithoutBias.addMatrices(biases);
      return result;
  }

  private static Matrix sigmoidFunction(Matrix zValues) {
      double[][] resultingValues = new double[zValues.rowSize][zValues.columnSize];

      for (int i = 0; i < zValues.rowSize; i++) {
          for (int j = 0; j < zValues.columnSize; j++) {
              double result = (1 / (1 + Math.pow(Math.E, -zValues.grid[i][j])));
              resultingValues[i][j] = result;
          }
      }

      Matrix result = new Matrix(resultingValues);
      return result;
  }

  public void calculateBiasGradientForFinalLayer(Matrix expectedOutputs) {
      Matrix subExpecteds = activationValues.subtractMatrices(expectedOutputs);
      
      // create the appropriately sized matrix of 1's
      double[][] ones = new double[activationValues.rowSize][activationValues.columnSize];

      for (int i = 0; i < activationValues.rowSize; i++) {
          for (int j = 0; j < activationValues.columnSize; j++) {
              ones[i][j] = 1.0;
          }
      }

      Matrix matrixOf1s = new Matrix(ones);
      Matrix onesMinusActivations = matrixOf1s.subtractMatrices(activationValues);
      Matrix hadamarOfRight = activationValues.hadamardProduct(onesMinusActivations);
      Matrix result = subExpecteds.hadamardProduct(hadamarOfRight);

      biasGradient = result;
  }

  // Note: (biasGradient / of right)
  public void calculateBiasGradient(Matrix weightsOfRight, Matrix biasesGradientOfRight) {
      // this represents the (Weights of l + 1 dotted with the biasGradient of l + 1)
      Matrix rightWeightsTranspose = weightsOfRight.transposeMatrix();
      Matrix dottedWeightsBiasGradientsOfRight = rightWeightsTranspose.dotProductMatrices(biasesGradientOfRight);

      double[][] ones = new double[activationValues.rowSize][activationValues.columnSize];

      for (int i = 0; i < activationValues.rowSize; i++) {
          for (int j = 0; j < activationValues.columnSize; j++) {
              ones[i][j] = 1.0;
          }
      }

      Matrix matrixOf1s = new Matrix(ones);
      Matrix onesMinusActivations = matrixOf1s.subtractMatrices(this.activationValues);
      // this is (activationMatrix) hadamarded (1's matrix - activationMatrix)
      Matrix rightSizeOfBiasGradientEquation = activationValues.hadamardProduct(onesMinusActivations);
      Matrix result = dottedWeightsBiasGradientsOfRight.hadamardProduct(rightSizeOfBiasGradientEquation);

      biasGradient = result;
  }

  public void calculateWeightGradient(Matrix activationsLeft) {
      Matrix activationTranspose = activationsLeft.transposeMatrix();
      Matrix result = biasGradient.dotProductMatrices(activationTranspose);
      weightGradient = result;
  }

}
