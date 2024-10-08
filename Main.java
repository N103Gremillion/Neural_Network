
class Main {

    public static void main(String args[]) {
        Matrix matrix = new Matrix(10, 10);
        Matrix matrix2 = new Matrix(10, 10);

        for (int i = 0; i < 10; i++) {
            System.out.println();
            for (int j = 0; j < 10; j++) {
                System.out.print(j);
                matrix.grid[i][j] = (float) j;
                matrix2.grid[i][j] = (float) j;
            }
        }

        Matrix added = matrix.addMatrices(matrix2);

        for (int i = 0; i < 10; i++) {
            System.out.println();
            for (int j = 0; j < 10; j++) {
              System.out.print(added.grid[i][j] + " ");
            }
        }

        System.out.println();
        System.out.println();
        System.out.println();

        Matrix dotted = matrix.dotProductMatrices(matrix2);

        for (int i = 0; i < dotted.rowSize; i++) {
            System.out.println();
            for (int j = 0; j < dotted.columnSize; j++) {
                System.out.print(dotted.grid[i][j] + " ");
            }
        }

    }
}
