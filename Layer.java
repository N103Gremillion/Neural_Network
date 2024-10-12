
/*************Class representing a perceptron*******************/
public class Layer {

  Matrix activationValues;
  Matrix zValues;
  Matrix weights;
  Matrix biases;
  int layer;

  // only for the opening layer
  public Layer(float[][] inputs, int layer){

    this.activationValues = new Matrix(inputs);
    this.layer = layer;
    
  }

  // for every layer but layer 0
  public Layer(Matrix inputs, Matrix weights, Matrix biases, int layer){

    this.weights = weights;
    this.biases = biases;
    this.zValues = calculateZvalues(inputs, weights, biases);
    this.activationValues = sigmoidFunction(zValues);
    this.layer = layer;

  }

  private static Matrix calculateZvalues(Matrix prevActivations, Matrix weights, Matrix biases){
    Matrix dottedMatrixWithoutBias = weights.dotProductMatrices(prevActivations);
    Matrix result = dottedMatrixWithoutBias.addMatrices(biases);

    return result;
  }

  private static Matrix sigmoidFunction(Matrix zValues){

    float[][] resultingValues = new float[zValues.rowSize][zValues.columnSize];

    for (int i = 0; i < zValues.rowSize; i++){
        for (int j = 0; j < zValues.columnSize; j++){
            double result = (1 / (1 + Math.pow(Math.E, -zValues.grid[i][j])));
            resultingValues[i][j] = (float) result;
        }
    }

    Matrix result = new Matrix(resultingValues);

    return result;
  } 
}