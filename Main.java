
class Main {

    public static void main(String args[]) {
        setupCase1();

    }

    // function to setup and test the inputs/weights/biases in the excel file (kinda the entry point for main to reference)
    public static void setupCase1(){

        int learningRate = 10;

        //******************* for training case # 1 *************
        int case1totalLayers = 3;
        float[][] layer0Input = {{0}, {1}, {0}, {1}};
        float[][] layer0ExpectedOutput = {{1}, {0}};

        //weights
        float[][] layer1Weights = {
            {-0.21f, 0.72f, -0.25f, 1.0f},
            {-0.94f, -0.41f, -0.47f, 0.63f},
            {0.15f, 0.55f, -0.49f, -0.75f}
        };
        float[][] layer2Weights = {
            {0.76f, 0.48f, -0.73f},
            {0.34f, 0.89f, -0.23f}
        };

        //biases
        float[][] layer1Biases = {{0.1f}, {-0.36f}, {-0.31f}};
        float[][] layer2Biases = {{0.16f}, {-0.46f}};

        // initiate the startingLayer
        NeuralNetwork network1 = new NeuralNetwork(layer0Input, 0, case1totalLayers);
        System.out.println("\nActivations for Layer 0");
        network1.layers[0].activationValues.printMatrix();
        
        // 1st forward pass
        network1.forwardPass(network1.layers[0].activationValues, new Matrix(layer1Weights), new Matrix(layer1Biases), 1); 
        System.out.println("\nActivations for Layer 1");
        network1.layers[1].activationValues.printMatrix();

        // 2nd forward pass
        network1.forwardPass(network1.layers[1].activationValues, new Matrix(layer2Weights), new Matrix(layer2Biases), 2); 
        System.out.println("\nActivations for Layer 2");
        network1.layers[2].activationValues.printMatrix();
    }
}
