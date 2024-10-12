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
        if (this.rowSize != matrix.rowSize || this.columnSize != matrix.columnSize) {
            return null;
        }

        float[][] resultingValues = new float[this.rowSize][this.columnSize];

        for (int row = 0; row < this.rowSize; row++) {
            for (int column = 0; column < this.columnSize; column++) {

                float resultingData = this.grid[row][column] + matrix.grid[row][column];
                resultingValues[row][column] = resultingData;

            }
        }

        Matrix resultingMatrix = new Matrix(resultingValues);

        return resultingMatrix;
    }

    // note : matrix 1 is being dotted by matrix 2 so matrix1 . matrix2
    public Matrix dotProductMatrices(Matrix matrix2) {

        // invalid demensions check
        if (this.columnSize != matrix2.rowSize) {
            return null;
        }

        float[][] resultingValues = new float[this.rowSize][matrix2.columnSize];

        for (int m1row = 0; m1row < this.rowSize; m1row++) {

            // iterate over the current row (matrix1) / to every column in matrix 2
            for (int m2col = 0; m2col < matrix2.columnSize; m2col++) {

                // var to keep track of the dot product of a row . column
                float rowColumnDottedData = 0;

                for (int m1col = 0; m1col < this.columnSize; m1col++) {

                    float curM1val = this.grid[m1row][m1col];
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

        if (this.rowSize != matrix2.rowSize || this.columnSize != matrix2.columnSize) {
            return null;
        }

        float[][] resultingValues = new float[this.rowSize][this.columnSize];

        for (int row = 0; row < this.rowSize; row++) {
            for (int column = 0; column < this.columnSize; column++) {

                float resultingData = this.grid[row][column] * matrix2.grid[row][column];
                resultingValues[row][column] = resultingData;

            }
        }

        Matrix resultingMatrix = new Matrix(resultingValues);

        return resultingMatrix;
    }

    public void printMatrix(){

        System.out.println("The Values for this Matrix are \n ****************************************");

        for (int i = 0; i < this.rowSize; i++){
            for (int j = 0; j < this.columnSize; j++){
                System.out.print(this.grid[i][j] + ", ");
            }
            System.out.println();
        }
    }

}
