import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

class Main {

    static double learningRate = 3.000;
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
        
        // train the network from the mnist_train.csv (uses 10 as the minibatach aka. 1 row of the trainingNetworks)
        System.out.println("layer 2 weights before 1st epoch");

        print2dArray(layer2biases);
        
        System.out.println("layer 2 weights after 1st epoch");

        print2dArray(layer1biases);

        runEpoch(1, trainingNetworks[0], expectedOutputsOfTrainingNetworks[0]);
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

                // if your hit the 30 epoch total
                if (curLine >= totalEpochs * miniBatchSize) {
                    break;
                }

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
    public static void updateUsingGradients(double learningRate, int numOfCases, double[][] oldValues, Matrix[] caseGradients){

        double learningRateOverNumOfCases = learningRate / numOfCases;
        Matrix gradientSum = caseGradients[0].addMatrices(caseGradients[1]);

        for (int i = 2; i < caseGradients.length; i++){
            gradientSum = gradientSum.addMatrices(caseGradients[i]);
        }

        Matrix scaledGradientSum =  gradientSum.scalarMultiply(learningRateOverNumOfCases);

        // update the old weights/biases
        for (int i = 0; i < oldValues.length; i++){
            for (int j = 0; j < oldValues[0].length; j++){
                oldValues[i][j] = (oldValues[i][j] - scaledGradientSum.grid[i][j]);
            }
        }
    }
    
    // function to setup and test the inputs/weights/biases in the excel file (kinda the entry point for main to reference)
    public static void runEpoch(int epochNum, NeuralNetwork[] networks, Matrix[] expecteds){
        
        // values to keep track of accuracy of the epoch
        int totalCorrectPredictions = 0;
        int totalPredictions = 0;
        int[] correctsForEachNum = new int[10];
        int[] totalForEachNum = new int[];

        //********************************Epoc 1***************************************** */
        for (int caseNum = 0; caseNum < networks.length; caseNum++) {
            NeuralNetwork currentNetwork = networks[caseNum];
            Matrix currentExpectedOutput = expecteds[caseNum];

            // Forward pass for both layers
            currentNetwork.forwardPass(currentNetwork.layers[0].activationValues, new Matrix(layer1weights), new Matrix(layer1biases), 1);
            currentNetwork.forwardPass(currentNetwork.layers[1].activationValues, new Matrix(layer2weights), new Matrix(layer2biases), 2);

            // update the cout of the num / correctly predicted num
            int predictedNum = 
            if (pr)
            // Backpropagation
            for (int curLayer = totalLayers; curLayer > 1; curLayer--) {
                currentNetwork.backwardPropogate(currentExpectedOutput, curLayer);
            }
        }
        /***********************adjust the weights after all 10 cases for this minibatch********************************/ 
        Matrix[] layer2WeightGradients = new Matrix[networks.length];
        Matrix[] layer1WeightGradients = new Matrix[networks.length];
        Matrix[] layer2BiasGradients = new Matrix[networks.length];
        Matrix[] layer1BiasGradients = new Matrix[networks.length];

        // Loop through each network and populate the arrays
        for (int i = 0; i < networks.length; i++) {
            layer2WeightGradients[i] = networks[i].layers[2].weightGradient;
            layer1WeightGradients[i] = networks[i].layers[1].weightGradient;
            layer2BiasGradients[i] = networks[i].layers[2].biasGradient;
            layer1BiasGradients[i] = networks[i].layers[1].biasGradient;
        }

        updateUsingGradients(learningRate, 10, layer2weights, layer2WeightGradients);
        updateUsingGradients(learningRate, 10, layer1weights, layer1WeightGradients);
        updateUsingGradients(learningRate, 10, layer2biases, layer2BiasGradients);
        updateUsingGradients(learningRate, 10, layer1biases, layer1BiasGradients);
        
        System.out.println("\nThe Update weights / biases after this minibatch are : \n");

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
