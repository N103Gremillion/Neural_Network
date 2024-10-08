/*********************** class to represent a matrix **********************/

class Matrix{

  // declarations
  int rowSize;
  int columnSize;
  float[][] grid;

  /***Constructor***/
  public Matrix(int rowSize, int columnSize){

    // definitions
    this.rowSize = rowSize;
    this.columnSize = columnSize;
    this.grid = new int[rowSize][columnSize];

  }

  /*****************Methods******************/
  // add matricies
  Matrix addMatricies(Matrix matrix1, Matrix matrix2){

    // invalid matrix additon check
    if (matrix1.rowSize != matrix2.rowSize || matrix1.columnSize != matrix2.columnSize){
      return null;
    }

    Matrix resultingMatrix = new Matrix(matrix1.rowSize, matrix1.columnSize);

    for (int row = 0; row < matrix1.rowSize; row++){
      for (int column = 0; column < matrix1.columnSize; column++){

          int resultingData = matrix1.grid[row][column] + matrix2.grid[row][column];
          resultingMatrix.grid[row][column] = resultingData;

      }
    }
    return resultingMatrix;  
  }

  // note : matrix 1 is being dotted by matrix 2 so matrix1 . matrix2
  Matrix dotProductMatricies(Matrix matrix1, Matrix matrix2){

    // invalid demensions check
    if (matrix1.columnSize != matrix2.rowSize){
      return null;
    }

    Matrix resultingMatrix = new Matrix(matrix1.rowSize, matrix2.columnSize);

    for (int m1row = 0; m1row < matrix1.rowSize; row++){

      // iterate over the current row (matrix1) / to every column in matrix 2
      for (int m2col = 0; m2col < matrix2.columnSize; m2col++){

        // var to keep track of the dot product of a row . column
        int rowColumnDottedData = 0;

        for (int m1col = 0; m1col < matrix1.columnSize; m1col++){

          int curM1val = matrix1.grid[m1row][m1col];
          int curM2val = matrix2.grid[m1column][m2col];
          RowColumnDottedData += (curM1val  * curM2val);

        }

        resultingMatrix.grid[m1row][m2col] = rowColumnDottedData;

      }
    }

    return resultingMatrix;

  }

}