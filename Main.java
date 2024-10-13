
class Main {

    public static void main(String args[]) {
        setupCase1();

    }

    // function to setup and test the inputs/weights/biases in the excel file (kinda the entry point for main to reference)
    public static void setupCase1(){

        int learningRate = 10;

        //********************************Epoc 1***************************************** */

        //******************* for training case # 1 *************
        int case1totalLayers = 3;
        float[][] layer0InputCase1 = {{0}, {1}, {0}, {1}};
        float[][] case1ExpectedOutput = {{0}, {1}};

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
        NeuralNetwork network1 = new NeuralNetwork(layer0InputCase1, 0, case1totalLayers);
        System.out.println("\n*************************Case 1****************************");
        
        // 1st forward pass
        network1.forwardPass(network1.layers[0].activationValues, new Matrix(layer1Weights), new Matrix(layer1Biases), 1); 

        // 2nd forward pass
        network1.forwardPass(network1.layers[1].activationValues, new Matrix(layer2Weights), new Matrix(layer2Biases), 2); 

        // back propogation
        for (int curLayer = case1totalLayers; curLayer > 1; curLayer--){
            network1.backwardPropogate(new Matrix(case1ExpectedOutput), curLayer);
        }
       
        /**********************training case # 2***************************/
        int case2totalLayers = 3;
        float[][] layer0InputCase2 = {{1}, {0}, {1}, {0}};
        float[][] case2ExpectedOutput = {{1}, {0}};

        // initiate the startingLayer
        NeuralNetwork network2 = new NeuralNetwork(layer0InputCase2, 0, case2totalLayers);
        
        // 1st forward pass
        network2.forwardPass(network2.layers[0].activationValues, new Matrix(layer1Weights), new Matrix(layer1Biases), 1); 
        network2.forwardPass(network2.layers[1].activationValues, new Matrix(layer2Weights), new Matrix(layer2Biases), 2);
        
        for (int curLayer = case1totalLayers; curLayer > 1; curLayer--){
            network2.backwardPropogate(new Matrix(case2ExpectedOutput), curLayer);
        }

        /***********************adjust the weights after case 1 and 2********************************/ 

    }
}
