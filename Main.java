import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

class Main {

    static double learningRate = 1;
    static int totalLayers = 3;
    static int totalEpochs = 60;
    static int miniBatchSize = 10;
    static int inputLinesSize = 785;
    static int totalDataPoints = 60000;
    static int totalMiniBatches = totalDataPoints / miniBatchSize;
    static Matrix layer1weights;
    static Matrix layer2weights;
    static Matrix layer1biases;
    static Matrix layer2biases;
    // this will hold all the csv data in a readable format for my setup
    static NeuralNetwork[][] trainingNetworks;
    // this corresponds to the trainingNetworks and has the expected values using the label of the 1st value in each csv row
    static Matrix[][] expectedOutputsOfTrainingNetworks;
    
    public static void main(String args[]) {
        
        // initalize the training network / expected outputs

        trainingNetworks = new  NeuralNetwork[totalMiniBatches][miniBatchSize];
        expectedOutputsOfTrainingNetworks = new Matrix[totalMiniBatches][miniBatchSize];

        readCSV("mnist_train.csv");
        setupInitialWeightsAndBiases();

        // train the network from the mnist_train.csv (uses 10 as the minibatach aka. 1 row of the trainingNetworks)
        for (int curEpoch = 1; curEpoch <= totalEpochs; curEpoch++){
            runEpoch(curEpoch, trainingNetworks, expectedOutputsOfTrainingNetworks);
        }

    }

    public static void setupInitialWeightsAndBiases(){
        // fill the wieghts with random values from -1 -> 1 (note : this is 15x784 since layer0 is 784 and layer 1 is 15)
        double[][] layer1weightsValues = initalizeWeightsOrBiases(15, 784);
        double[][] layer2weightsValues = initalizeWeightsOrBiases(10, 15);
        double[][] layer1biasesValues = initalizeWeightsOrBiases(15, 1);
        double[][] layer2biasesValues = initalizeWeightsOrBiases(10, 1);

        layer1weights = new Matrix(layer1weightsValues);
        layer2weights = new Matrix(layer2weightsValues);
        layer1biases = new Matrix(layer1biasesValues);
        layer2biases = new Matrix(layer2biasesValues);
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
                if (curLine >= 60000) {
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
            double scaledValue = curIntValue / 255.0;
            trainingNetworkInputs[i - 1][0] = scaledValue;
        }

        NeuralNetwork trainingNetwork = new NeuralNetwork(trainingNetworkInputs, 0, totalLayers);
        trainingNetworks[row][column] = trainingNetwork;

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

    public static void print2dArray(double[][] array2d) {
        // Print the 2D array
        for (double[] row : array2d) {
            for (double value : row) {
                System.out.print(value + " ");
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
    
    // use the output activation and find the one with largest probablity
    private static int getPredictedOutput(Matrix outputLayer) {
        
        int curPrediction = 0;
        double highestProbability = outputLayer.grid[0][0];

        for (int i = 0; i < outputLayer.grid.length; i++){
            double curProbability = outputLayer.grid[i][0];
            if (curProbability > highestProbability){
                highestProbability = curProbability;
                curPrediction = i;
            }
        }
        
        return curPrediction;

    }

    // convert the 1 hot value to int representation
    private static int getIntOfHotValue(Matrix currentExpectedOutput){
        int result = 0;

        for (int i = 0; i < currentExpectedOutput.grid.length; i++){
            if (currentExpectedOutput.grid[i][0] ==  1){
                result = i;
                break;
            }
        }

        return result;
    }

    // function to setup and test the inputs/weights/biases in the excel file (kinda the entry point for main to reference)
    public static void runEpoch(int epochNum, NeuralNetwork[][] networks, Matrix[][] expecteds) {
        int totalCorrectPredictions = 0;
        int totalPredictions = 0;
        int[] correctsForEachNum = new int[10];
        int[] totalForEachNum = new int[10];

        System.out.println(String.format("************************************ EPOCH %d **********************************************", epochNum));

        for (int curRow = 0; curRow < networks.length; curRow++) {
            for (int caseNum = 0; caseNum < miniBatchSize; caseNum++) {
                NeuralNetwork currentNetwork = networks[curRow][caseNum];
                Matrix currentExpectedOutput = expecteds[curRow][caseNum];

                // Forward pass
                currentNetwork.forwardPass(currentNetwork.layers[0].activationValues, layer1weights, layer1biases, 1);
                currentNetwork.forwardPass(currentNetwork.layers[1].activationValues, layer2weights, layer2biases, 2);

                // Get predictions 
                int predictedOutput = getPredictedOutput(currentNetwork.layers[2].activationValues);
                int expectedIntegerOutput = getIntOfHotValue(currentExpectedOutput);

                // Backpropagation
                for (int curLayer = totalLayers; curLayer > 1; curLayer--) {
                    currentNetwork.backwardPropogate(currentExpectedOutput, curLayer);
                }

                if (predictedOutput == expectedIntegerOutput) {
                    totalCorrectPredictions++;
                    correctsForEachNum[predictedOutput]++;
                }
                totalPredictions++;
                totalForEachNum[expectedIntegerOutput]++;
            }

            // Adjust the weights after processing the mini-batch
            updateWeightsForMiniBatch(networks[curRow]);
        }

        // Display epoch accuracy
        double accuracy = (double) totalCorrectPredictions / totalPredictions * 100;
        System.out.printf("Total accuracy of Epoch %d: %d / %d (%.2f%%)%n", epochNum, totalCorrectPredictions, totalPredictions, accuracy);
        // accuracies for each of the numbers
        for (int i = 0; i < correctsForEachNum.length; i++){
            double curNumAccuracy = (double) correctsForEachNum[i] / totalForEachNum[i] * 100;
            System.out.println(String.format("The accuracy of %d : %d / %d or %.2f", i, correctsForEachNum[i], totalForEachNum[i], curNumAccuracy));
        }

        System.out.println(String.format("************************************ End of Epoch %d **********************************************", epochNum));
    }


    /**
     * Update weights and biases using gradients collected from a mini-batch of networks.
     */
    private static void updateWeightsForMiniBatch(NeuralNetwork[] miniBatch) {
        Matrix[] layer2WeightGradients = new Matrix[miniBatch.length];
        Matrix[] layer1WeightGradients = new Matrix[miniBatch.length];
        Matrix[] layer2BiasGradients = new Matrix[miniBatch.length];
        Matrix[] layer1BiasGradients = new Matrix[miniBatch.length];

        // Collect gradients from each network in the mini-batch
        for (int i = 0; i < miniBatch.length; i++) {
            layer2WeightGradients[i] = miniBatch[i].layers[2].weightGradient;
            layer1WeightGradients[i] = miniBatch[i].layers[1].weightGradient;
            layer2BiasGradients[i] = miniBatch[i].layers[2].biasGradient;
            layer1BiasGradients[i] = miniBatch[i].layers[1].biasGradient;
        }

        // Update weights and biases
        updateUsingGradients(learningRate, miniBatch.length, layer2weights.grid, layer2WeightGradients);
        updateUsingGradients(learningRate, miniBatch.length, layer1weights.grid, layer1WeightGradients);
        updateUsingGradients(learningRate, miniBatch.length, layer2biases.grid, layer2BiasGradients);
        updateUsingGradients(learningRate, miniBatch.length, layer1biases.grid, layer1BiasGradients);
    }

}
