
class Main {

    public static void main(String args[]) {
        setupNetwork();

    }

    // function to setup and test the inputs/weights/biases in the excel file (kinda the entry point for main to reference)
    public static void setupNetwork(){
        float[][] inputLayer1 = {{1}, {0}, {1}, {0}};
        float[][] inputWeights = {
           {-0.21f, 0.72f, -0.25f, 1.0f},
            {-0.94f, -0.41f, -0.47f, 0.63f},
            {0.15f, 0.55f, -0.49f, -0.75f}
        };
    }
}
