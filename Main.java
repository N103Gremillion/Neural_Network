
class Main {

    public static void main(String args[]) {
        
        // Example input values and weights
        float[][] inputValues = {{0.5f, 0.3f}};
        float[][] weights = {{0.2f}, {0.8f}};
        float bias = 0.1f;

        // Create a new Perceptron
        Perceptron perceptron = new Perceptron(inputValues, weights, bias);

        // Print out the zValue and activationValue
        System.out.println("zValue: " + perceptron.zValue);
        System.out.println("Activation Value: " + perceptron.activationValue);

    }
}
