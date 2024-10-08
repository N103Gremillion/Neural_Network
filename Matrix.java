
/** ********************* class to represent a matrix ********************* */
class Matrix {

    // declarations
    int rowSize;
    int columnSize;
    float[][] grid;

    /**
     * *Constructor**
     */
    public Matrix(int rowSize, int columnSize) {

        // definitions
        this.rowSize = rowSize;
        this.columnSize = columnSize;
        this.grid = new float[rowSize][columnSize];

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

        Matrix resultingMatrix = new Matrix(this.rowSize, this.columnSize);

        for (int row = 0; row < this.rowSize; row++) {
            for (int column = 0; column < this.columnSize; column++) {

                float resultingData = this.grid[row][column] + matrix.grid[row][column];
                resultingMatrix.grid[row][column] = resultingData;

            }
        }
        return resultingMatrix;
    }

    // note : matrix 1 is being dotted by matrix 2 so matrix1 . matrix2
    public Matrix dotProductMatrices(Matrix matrix2) {

        // invalid demensions check
        if (this.columnSize != matrix2.rowSize) {
            return null;
        }

        Matrix resultingMatrix = new Matrix(this.rowSize, matrix2.columnSize);

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

                resultingMatrix.grid[m1row][m2col] = rowColumnDottedData;

            }
        }

        return resultingMatrix;

    }

}
