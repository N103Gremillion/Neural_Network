
class Main {

    static float[][] layer1weights;
    static float[][] layer2weights;
    static float[][] layer1biases;
    static float[][] layer2biases;
    static int learningRate = 10;
    static int totalLayers = 3;

    public static void main(String args[]) {

        setupInitialWeights();
        setupInitialBiases();
        runEpoch(1);
        runEpoch(2);
        runEpoch(3);
        runEpoch(4);
        runEpoch(5);
        runEpoch(6);
        
    }

    public static void setupInitialWeights(){
        //weights
        layer1weights = new float[][] {
            {-0.21f, 0.72f, -0.25f, 1.0f},
            {-0.94f, -0.41f, -0.47f, 0.63f},
            {0.15f, 0.55f, -0.49f, -0.75f}
        };

        layer2weights = new float[][] {
            {0.76f, 0.48f, -0.73f},
            {0.34f, 0.89f, -0.23f}
        };
    }

    public static void setupInitialBiases(){
        //biases
        layer1biases = new float[][] {{0.1f}, {-0.36f}, {-0.31f}};
        layer2biases = new float[][] {{0.16f}, {-0.46f}};
    }

    public static void print2dArray(float[][] array2d){

        for (int i = 0; i < array2d.length; i++) { 

            for (int j = 0; j < array2d[i].length; j++) {  
                System.out.print(array2d[i][j] + " ");  
            }

        System.out.println();  
        }

    }
    // note it is assumed that the networks are the same sizes but differenct cases
    public static void updateUsingGradients(int learningRate, int numOfCases, float[][] oldValues, Matrix network1Gradient, Matrix network2Gradient){

        float learningRateOverNumOfCases = learningRate / numOfCases;
        Matrix gradientSum = network1Gradient.addMatrices(network2Gradient);
        Matrix scaledGradientSum =  gradientSum.scalarMultiply(learningRateOverNumOfCases);

        // update the old weights/biases
        for (int i = 0; i < oldValues.length; i++){
            for (int j = 0; j < oldValues[0].length; j++){
                oldValues[i][j] = (oldValues[i][j] - scaledGradientSum.grid[i][j]);
            }
        }
    }
    
    // function to setup and test the inputs/weights/biases in the excel file (kinda the entry point for main to reference)
    public static void runEpoch(int epochNum){
        //********************************Epoc 1***************************************** */
        //******************* for training case # 1 *************
        float[][] layer0InputCase1 = {{0}, {1}, {0}, {1}};
        float[][] case1ExpectedOutput = {{0}, {1}};

        // initiate the startingLayer
        NeuralNetwork network1 = new NeuralNetwork(layer0InputCase1, 0, totalLayers);
        System.out.println(String.format("************************* Epoch %d ************************************", epochNum));
        System.out.println("\n*************************Case 1 / Case 2****************************");
        
        // 1st forward pass
        network1.forwardPass(network1.layers[0].activationValues, new Matrix(layer1weights), new Matrix(layer1biases), 1); 

        // 2nd forward pass
        network1.forwardPass(network1.layers[1].activationValues, new Matrix(layer2weights), new Matrix(layer2biases), 2); 

        // back propogation
        for (int curLayer = totalLayers; curLayer > 1; curLayer--){
            network1.backwardPropogate(new Matrix(case1ExpectedOutput), curLayer);
        }
       
        /**********************training case # 2***************************/
        float[][] layer0InputCase2 = {{1}, {0}, {1}, {0}};
        float[][] case2ExpectedOutput = {{1}, {0}};

        // initiate the startingLayer
        NeuralNetwork network2 = new NeuralNetwork(layer0InputCase2, 0, totalLayers);
        
        // 1st forward pass
        network2.forwardPass(network2.layers[0].activationValues, new Matrix(layer1weights), new Matrix(layer1biases), 1); 
        network2.forwardPass(network2.layers[1].activationValues, new Matrix(layer2weights), new Matrix(layer2biases), 2);
        
        for (int curLayer = totalLayers; curLayer > 1; curLayer--){
            network2.backwardPropogate(new Matrix(case2ExpectedOutput), curLayer);
        }

        /***********************adjust the weights after case 1 and 2********************************/ 
        updateUsingGradients(learningRate, 2, layer2weights, network1.layers[2].weightGradient, network2.layers[2].weightGradient);
        updateUsingGradients(learningRate, 2, layer1weights, network1.layers[1].weightGradient, network2.layers[1].weightGradient);
        updateUsingGradients(learningRate, 2, layer2biases, network1.layers[2].biasGradient, network2.layers[2].biasGradient);
        updateUsingGradients(learningRate, 2, layer1biases, network1.layers[1].biasGradient, network2.layers[1].biasGradient);
        
        System.out.println("\nThe Update weights / biases after the first 2 cases are : \n");

        System.out.println("\n************** Layer 1 Weights ******************\n");
        print2dArray(layer1weights);
        System.out.println("\n************** Layer 2 Weights ******************\n");
        print2dArray(layer2weights);
        System.out.println("\n************** Layer 1 Biases ******************\n");
        print2dArray(layer1biases);
        System.out.println("\n************** Layer 2 Biases ******************\n");
        print2dArray(layer2biases);

        /*****************training case # 3************************ */
        float[][] layer0InputCase3 = {{0}, {0}, {1}, {1}};
        float[][] case3ExpectedOutput = {{0}, {1}};

        NeuralNetwork network3 = new NeuralNetwork(layer0InputCase3, 0, totalLayers);
        System.out.println("\n*************************Case 3 / Case 4****************************");
        
        // 1st forward pass
        network3.forwardPass(network3.layers[0].activationValues, new Matrix(layer1weights), new Matrix(layer1biases), 1); 
        // 2nd forward pass
        network3.forwardPass(network3.layers[1].activationValues, new Matrix(layer2weights), new Matrix(layer2biases), 2); 

        // back propogation
        for (int curLayer = totalLayers; curLayer > 1; curLayer--){
            network3.backwardPropogate(new Matrix(case3ExpectedOutput), curLayer);
        }
       
        /**********************training case # 4***************************/
        float[][] layer0InputCase4 = {{1}, {1}, {0}, {0}};
        float[][] case4ExpectedOutput = {{1}, {0}};

        // initiate the startingLayer
        NeuralNetwork network4 = new NeuralNetwork(layer0InputCase4, 0, totalLayers);
        
        // 1st forward pass
        network4.forwardPass(network4.layers[0].activationValues, new Matrix(layer1weights), new Matrix(layer1biases), 1); 
        network4.forwardPass(network4.layers[1].activationValues, new Matrix(layer2weights), new Matrix(layer2biases), 2);
        
        for (int curLayer = totalLayers; curLayer > 1; curLayer--){
            network4.backwardPropogate(new Matrix(case4ExpectedOutput), curLayer);
        }

        /***********************adjust the weights after case 1 and 2********************************/ 
        updateUsingGradients(learningRate, 2, layer2weights, network4.layers[2].weightGradient, network3.layers[2].weightGradient);
        updateUsingGradients(learningRate, 2, layer1weights, network4.layers[1].weightGradient, network3.layers[1].weightGradient);
        updateUsingGradients(learningRate, 2, layer2biases, network4.layers[2].biasGradient, network3.layers[2].biasGradient);
        updateUsingGradients(learningRate, 2, layer1biases, network4.layers[1].biasGradient, network3.layers[1].biasGradient);
        
        System.out.println("\nThe Update weights / biases after the second 2 cases are : \n");

        System.out.println("\n************** Layer 1 Weights ******************\n");
        print2dArray(layer1weights);
        System.out.println("\n************** Layer 2 Weights ******************\n");
        print2dArray(layer2weights);
        System.out.println("\n************** Layer 1 Biases ******************\n");
        print2dArray(layer1biases);
        System.out.println("\n************** Layer 2 Biases ******************\n");
        print2dArray(layer2biases);
        System.out.println(String.format("\n******************************** End of Epoc %d ***********************************", epochNum));
    }

    
}
