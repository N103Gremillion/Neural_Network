/** ********************* class to represent a matrix ********************* */
class Matrix {

    // declarations
    int rowSize;
    int columnSize;
    float[][] grid;

    /**
     * *Constructor**
     */
    public Matrix(float[][] inputValues) {
        if (inputValues == null || inputValues.length == 0 || inputValues[0].length == 0) {
            System.out.println("invalid input values");
        }

        this.rowSize = inputValues.length;
        this.columnSize = inputValues[0].length;
        
        // Deep copy of inputValues to avoid external modifications affecting the grid
        this.grid = new float[rowSize][columnSize];
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < columnSize; j++){
                grid[i][j] = inputValues[i][j];
            }
        }
    }

    /**
     * ***************Methods*****************
     */

    // add matricies
    public Matrix addMatrices(Matrix matrix) {

        // invalid matrix additon check
        if (rowSize != matrix.rowSize || columnSize != matrix.columnSize) {
            return null;
        }

        float[][] resultingValues = new float[rowSize][columnSize];

        for (int row = 0; row < rowSize; row++) {
            for (int column = 0; column < columnSize; column++) {

                float resultingData = grid[row][column] + matrix.grid[row][column];
                resultingValues[row][column] = resultingData;

            }
        }

        Matrix resultingMatrix = new Matrix(resultingValues);

        return resultingMatrix;
    }


    public Matrix subtractMatrices(Matrix matrix) {

        // invalid matrix additon check
        if (rowSize != matrix.rowSize || columnSize != matrix.columnSize) {
            return null;
        }

        float[][] resultingValues = new float[rowSize][columnSize];

        for (int row = 0; row < rowSize; row++) {
            for (int column = 0; column < columnSize; column++) {

                float resultingData = grid[row][column] - matrix.grid[row][column];
                resultingValues[row][column] = resultingData;

            }
        }

        Matrix resultingMatrix = new Matrix(resultingValues);

        return resultingMatrix;
    }

    // note : matrix 1 is being dotted by matrix 2 so matrix1 . matrix2
    public Matrix dotProductMatrices(Matrix matrix2) {

        // invalid demensions check
        if (columnSize != matrix2.rowSize) {
            return null;
        }

        float[][] resultingValues = new float[rowSize][matrix2.columnSize];

        for (int m1row = 0; m1row < rowSize; m1row++) {

            // iterate over the current row (matrix1) / to every column in matrix 2
            for (int m2col = 0; m2col < matrix2.columnSize; m2col++) {

                // var to keep track of the dot product of a row . column
                float rowColumnDottedData = 0;

                for (int m1col = 0; m1col < columnSize; m1col++) {

                    float curM1val = grid[m1row][m1col];
                    float curM2val = matrix2.grid[m1col][m2col];
                    rowColumnDottedData += (curM1val * curM2val);
                }

                resultingValues[m1row][m2col] = rowColumnDottedData;

            }
        }

        Matrix resultingMatrix = new Matrix(resultingValues);

        return resultingMatrix;

    }

    public Matrix hadamardProduct(Matrix matrix2){

        if (rowSize != matrix2.rowSize || columnSize != matrix2.columnSize) {
            return null;
        }

        float[][] resultingValues = new float[rowSize][columnSize];

        for (int row = 0; row < rowSize; row++) {
            for (int column = 0; column < columnSize; column++) {

                float resultingData = grid[row][column] * matrix2.grid[row][column];
                resultingValues[row][column] = resultingData;

            }
        }

        Matrix resultingMatrix = new Matrix(resultingValues);

        return resultingMatrix;
    }

    public Matrix transposeMatrix(){
        float[][] result = new float[columnSize][rowSize];

        for (int i = 0; i < rowSize; i++){
            for (int j = 0; j < columnSize; j++){
                result[j][i] = grid[i][j];
            }
        }

        return new Matrix(result);
    }

    public Matrix scalarMultiply(float scalarValue){

        float[][] scaledGrid = new float[rowSize][columnSize];

        for (int i = 0; i < rowSize; i++){
            for (int j = 0; j < columnSize; j++){
                scaledGrid[i][j] = (grid[i][j] * scalarValue);
            }
        }

        Matrix result = new Matrix(scaledGrid);

        return result;
    }

    public void printMatrix(){

        System.out.println("****************************************");

        for (int i = 0; i < rowSize; i++){
            for (int j = 0; j < columnSize; j++){
                System.out.print(grid[i][j] + ", ");
            }
            System.out.println();
        }
    }

}
