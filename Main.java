import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

class Main {

    static int learningRate = 3;
    static int totalLayers = 3;
    static int totalEpochs = 30;
    static int miniBatchSize = 10;
    static int inputLinesSize = 785;
    static double[][] layer1weights;
    static double[][] layer2weights;
    static double[][] layer1biases;
    static double[][] layer2biases;
    // this will hold all the csv data in a readable format for my setup
    static NeuralNetwork[][] trainingNetworks = new NeuralNetwork[totalEpochs][miniBatchSize];
    // this corresponds to the trainingNetworks and has the expected values using the label of the 1st value in each csv row
    static Matrix[][] expectedOutputsOfTrainingNetworks = new Matrix[totalEpochs][miniBatchSize];
    
    public static void main(String args[]) {

        readCSV("mnist_train.csv");
        setupInitialWeightsAndBiases();
 
    }

    public static void setupInitialWeightsAndBiases(){
        // fill the wieghts with random values from -1 -> 1 (note : this is 15x784 since layer0 is 784 and layer 1 is 15)
        layer1weights = initalizeWeightsOrBiases(15, 784);
        layer2weights = initalizeWeightsOrBiases(10, 15);
        layer1biases = initalizeWeightsOrBiases(15, 1);
        layer2biases = initalizeWeightsOrBiases(10, 1);
    }

    // fill a 2d array with vlalues form -1 to 1
    private static double[][] initalizeWeightsOrBiases(int rows, int cols){
        double[][] values = new double[rows][cols];
        Random random = new Random();

        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[i].length; j++) {
                double randomVal = (random.nextDouble() * 2) - 1; 
                values[i][j] = randomVal; 
            }
        }
        return values;
    }

    /*  fills in the layer 0 for the traning networks and only reads up to epochSize * minibatchSize 
     *  Also fills in the expected output by getting the 1st value out of the row in the csv and turing
     *  it into a red hot vector
    */
    public static void readCSV(String filePath) {

        // try to read current filePath file
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            int curLine = 0;
            String line;

            // While there is a line to read from
            while ((line = reader.readLine()) != null) {

                String[] values = line.split(",");
                inputTrainingNetwork(values, curLine);

                curLine++;
            }
        }

        catch (IOException error) {
            error.printStackTrace();
        }
    
    }

    // adds a training network to the 2d array of networks also adds its coresponding expected value (note expects the network to match the demensions of the 1st layer of weights)
    public static void inputTrainingNetwork(String[] networkInputs, int lineNumber) {

        if (networkInputs == null || networkInputs.length <= 0){
            return;
        }

        // find the correct position in the trainingNetworks[][]/expectedOutputs[][] to put the newLine data
        int row = lineNumber / miniBatchSize;
        int column = lineNumber % miniBatchSize;

        // get the expected hot value from the 1st value in the array
        int integerExpected = Integer.parseInt(networkInputs[0]);
        Matrix oneHotVal = generateHotValue(integerExpected);
        expectedOutputsOfTrainingNetworks[row][column] = oneHotVal;

        // loop through the rest of the values to fill in the networks inputs (I hardCoded the inputLinesSize at the top)
        double[][] trainingNetworkInputs = new double[inputLinesSize - 1][1];
        for (int i = 1; i < inputLinesSize; i++){
            int curIntValue = Integer.parseInt(networkInputs[i]);
            trainingNetworkInputs[i - 1][0] = (double) curIntValue;
        }
        NeuralNetwork trainingNetwork = new NeuralNetwork(trainingNetworkInputs, 0, totalLayers);
    }

    // returns the oneHotValue of the integer that represents the expected output
    public static Matrix generateHotValue(int integerValue){

        if (integerValue < 0 || integerValue > 9){
            return null;
        }

        double[][] values = new double[10][1];

        for (int i = 0; i< values.length; i++){
            if (i == integerValue){
                values[i][0] = 1.00; 
            }
            else{
                values[i][0] = 0.00;
            }
        }

        return new Matrix(values);
    }

    public static void print2dArray(double[][] array2d){

        for (int i = 0; i < array2d.length; i++) { 

            for (int j = 0; j < array2d[i].length; j++) {  
                System.out.print(array2d[i][j] + " ");  
            }

        System.out.println();  
        }

    }

    // note it is assumed that the networks are the same sizes but different cases
    public static void updateUsingGradients(int learningRate, int numOfCases, double[][] oldValues, Matrix network1Gradient, Matrix network2Gradient){

        double learningRateOverNumOfCases = learningRate / numOfCases;
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
        double[][] layer0InputCase1 = {{0}, {1}, {0}, {1}};
        double[][] case1ExpectedOutput = {{0}, {1}};

        // initiate the startingLayer
        NeuralNetwork network1 = new NeuralNetwork(layer0InputCase1, 0, totalLayers);
        System.out.println(String.format("************************* Epoch %d ************************************", epochNum));
        System.out.println("\n*************************Case 1 / Case 2****************************");
        
        // 1st forward pass
        network1.forwardPass(network1.layers[0].activationValues, new Matrix(layer1weights), new Matrix(layer1biases), 1); 

        // 2nd forward pass
        network1.forwardPass(network1.layers[1].activationValues, new Matrix(layer2weights), new Matrix(layer2biases), 2); 

        // back propagation
        for (int curLayer = totalLayers; curLayer > 1; curLayer--){
            network1.backwardPropogate(new Matrix(case1ExpectedOutput), curLayer);
        }
       
        /**********************training case # 2***************************/
        double[][] layer0InputCase2 = {{1}, {0}, {1}, {0}};
        double[][] case2ExpectedOutput = {{1}, {0}};

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
        double[][] layer0InputCase3 = {{0}, {0}, {1}, {1}};
        double[][] case3ExpectedOutput = {{0}, {1}};

        NeuralNetwork network3 = new NeuralNetwork(layer0InputCase3, 0, totalLayers);
        System.out.println("\n*************************Case 3 / Case 4****************************");
        
        // 1st forward pass
        network3.forwardPass(network3.layers[0].activationValues, new Matrix(layer1weights), new Matrix(layer1biases), 1); 
        // 2nd forward pass
        network3.forwardPass(network3.layers[1].activationValues, new Matrix(layer2weights), new Matrix(layer2biases), 2); 

        // back propagation
        for (int curLayer = totalLayers; curLayer > 1; curLayer--){
            network3.backwardPropogate(new Matrix(case3ExpectedOutput), curLayer);
        }
       
        /**********************training case # 4***************************/
        double[][] layer0InputCase4 = {{1}, {1}, {0}, {0}};
        double[][] case4ExpectedOutput = {{1}, {0}};

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
    }
}
