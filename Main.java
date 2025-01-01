
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

class Calculas {

    public double[] weightedSum(double[] inputLayer, double[][] weight, double[][] bias) {
        double[] result = new double[bias.length];
        /*
        Rows represent the number of neurons in the next layer (the layer being connected to).
        Columns represent the number of neurons in the current layer (the layer providing the input).

        ex: for weights connection from input layer to hidden layer [15x784] say hidden layer size 15.
         */
        for (int i = 0; i < weight.length; i++) { // for each neurons in hidden layer we have biases
            result[i] = bias[i][0]; // since a col vector
            for (int j = 0; j < inputLayer.length; j++) { // all 784 neurons
                result[i] += weight[i][j] * inputLayer[j];
            }
        }
        return result;
    }

    public double sigmoid(double z) { // smooth version of step function, range between 0 to 1, 
        return 1 / (1 + Math.exp(-z));
    }

    public double sigmoidPrime(double z) {
        return sigmoid(z) * (1 - sigmoid(z));
    }

    public double[] sigmoidList(double[] z) {
        double[] activated = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            activated[i] = sigmoid(z[i]);
        }
        return activated;
    }

    public double[] sigmoidPrimeList(double[] z) {
        double[] sigma = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            sigma[i] = sigmoidPrime(z[i]);
        }
        return sigma;
    }

    public double[] deltaMse(double[] outputActivation, double[] y) {
        if (outputActivation.length != y.length) {
            throw new IllegalArgumentException("Arrays must have the same length.");
        }
        double[] errorList = new double[y.length];
        for (int i = 0; i < outputActivation.length; i++) {
            errorList[i] = outputActivation[i] - y[i];
        }
        return errorList;
    }

    public double[] oneHotEncode(int y, int index) {
        double[] newArr = new double[y];
        newArr[index] = 1;
        return newArr;
    }

    // Schur product
    public double[] schurProduct(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Arrays must have the same length.");
        }
        double[] product = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            product[i] = a[i] * b[i];
        }
        return product;
    }

    public double[][] schurProduct(double[][] matrixA, double[][] matrixB) {
        if (matrixA.length != matrixB.length || matrixA[0].length != matrixB[0].length) {
            throw new IllegalArgumentException("Arrays must have the same length.");
        }
        double[][] result = new double[matrixA.length][matrixA[0].length];
        for (int i = 0; i < matrixA.length; i++) {
            for (int j = 0; j < matrixA[0].length; j++) {
                result[i][j] = matrixA[i][j] * matrixB[i][j];
            }
        }

        return result;
    }

    public double[][] transpose(double[][] mat) {
        double[][] newDim = new double[mat[0].length][mat.length];
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                newDim[j][i] = mat[i][j];
            }
        }
        return newDim;
    }

    public double[][] outerProduct(double[][] a, double[][] b) {
        double[][] product = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                product[i][j] = a[i][0] * b[j][0];
            }
        }
        return product;
    }

    public double[][] dot(double[][] A, double[][] B) {
        int m = A.length;
        int n = B[0].length;
        int p = A[0].length;

        double[][] result = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < p; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }
}

class WeightBias {

    double[][][] bias;
    double[][][] weight;

    public double[][][] getBias() {
        return this.bias;
    }

    public double[][][] getWeight() {
        return this.weight;
    }

    public void setBias(double[][][] bias) {
        this.bias = bias;
    }

    public void setWeight(double[][][] weight) {
        this.weight = weight;
    }
}

class InputAndAcutalOutput {

    double[][] x;
    int[] y;
    int imageLength;

    public int getDataLength() {
        return this.imageLength;
    }

    public void setImageLength(int img) {
        this.imageLength = img;
    }

    public double[][] getInput() {
        return this.x;
    }

    public int[] getLabelOutput() {
        return this.y;
    }

    public void setInput(double[][] x) {
        this.x = x;
    }

    public void setLabels(int[] y) {
        this.y = y;
    }
}

class NeuralNetwork {

    Calculas calc;
    final int outputSize = 10;
    int hiddenLayerSize;
    final int neuralNetworkSize = 3;

    long startTime;

    List<double[][]> weights = new ArrayList<>(); // list of 2d array, ex: 0 correcponds to weights from input to hidden
    List<double[][]> biases = new ArrayList<>(); // list of 1d array, 0 corresponds to bias in hidden layer

    NeuralNetwork(int sizeOfHiddenLayer) {
        startTime = System.nanoTime();
        this.calc = new Calculas();
        // lets initilize random weight and biases
        // lets initlize random wieghts, and biases
        // each of the 784 pixel is connected to 10 neurons in the output layers
        this.hiddenLayerSize = sizeOfHiddenLayer;
        double[][] weightInput = initializeWeights(sizeOfHiddenLayer, 784); // input to hidden
        double[][] weightOutput = initializeWeights(outputSize, sizeOfHiddenLayer); // hidden to output

        // we need to initilize it as a col vector, if not then it will be 
        // problem when dealing with opreation finding delta for layer
        double[][] bias_hidden = initializeBiases(sizeOfHiddenLayer, 1);
        double[][] bias_output = initializeBiases(outputSize, 1);

        weights.add(weightInput);
        weights.add(weightOutput);

        biases.add(bias_hidden);
        biases.add(bias_output);

    }

    public double[][] initializeBiases(int rows, int cols) {
        double[][] b = new double[rows][cols];
        double variance = 1.0 / rows;
        double stdDev = Math.sqrt(variance);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                b[i][j] = random.nextGaussian() * stdDev;
            }
        }

        return b;
    }

    public double[][] initializeWeights(int rows, int cols) {
        double[][] w = new double[rows][cols];
        double variance = 1.0 / rows;
        double stdDev = Math.sqrt(variance);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                w[i][j] = random.nextGaussian() * stdDev;
            }
        }
        return w;
    }

    private double[][] rowVector(double[] vector) {
        // ex [1x15]
        double[][] newVector = new double[][]{vector};
        return newVector;
    }

    private double[][] colVector(double[] vector) {
        double[][] newVector = new double[][]{vector};
        return calc.transpose(newVector);
    }

    private double[] feedForward(double[] inputLayer) {
        double[] weightedSumHidden = calc.sigmoidList(calc.weightedSum(inputLayer, weights.get(0), biases.get(0)));
        double[] y = calc.sigmoidList(calc.weightedSum(weightedSumHidden, weights.get(1), biases.get(1)));
        return y;
    }

    //                       suffled input          
    private WeightBias train(double[] inputLayer, double[] y) {
        // forward pass, predicted values
        double[] weightedSumHidden = calc.weightedSum(inputLayer, weights.get(0), biases.get(0));
        double[] weightedSumOutput = calc.weightedSum(calc.sigmoidList(weightedSumHidden), weights.get(1), biases.get(1));
        /* my suspect is weighted sum 🤔
         * without activation gets extremely large over time
         * and gradient evuntally get vanished so,
         * x -> infinity then limit approaches to 1
         * x -> -ve infinity then limit apporaches to 0
         * there is a problem on how we calculate gradient,
         * and my guess was right, i forgot to activate
         */
        double[][][] nabla_weight = new double[neuralNetworkSize - 1][][];
        double[][][] nabla_bias = new double[neuralNetworkSize - 1][][];
        // for output layer
        double[] deltaC = calc.deltaMse(calc.sigmoidList(weightedSumOutput), y);
        // gradient of the cost function C wrt weight and biases respectively
        double[][] nabla_b = calc.schurProduct(colVector(deltaC), colVector(calc.sigmoidPrimeList(weightedSumOutput)));
        nabla_bias[nabla_bias.length - 1] = nabla_b; // [10 x 1] col vector
        // ∂C/∂b = delta; ∂C/∂w = delta . (a ^(l - 1))^T in our case activation in middel (hidden) layer
        // ex: expected dimension is [15x10] for output, and for hidden is [784x15] 
        // activation are stored as col vector
        double[][] nabla_w = calc.dot(nabla_b, rowVector(calc.sigmoidList(weightedSumHidden))); // activated layers  
        nabla_weight[nabla_weight.length - 1] = nabla_w; // bitch
        // back propragrte the error, hidden layer, nice practise we can hav multiple hidden layer
        // lets just understand that we are inside of hidden layer
        for (int i = 2; i >= 2; i--) {
            // we need z vectors of hidden layer
            // 🧩 I AM OFICALLY LOST 
            // col vector [10x1], in std way delta b = col vecotr
            double[][] transpoedWeight = calc.transpose(weights.get(1)); // wt of output, [10 x 15]^T = [15 x  10]
            double[][] weightTimesDelta = calc.dot(transpoedWeight, nabla_b); // [1x10] vector
            // activation in current layer aslo as a comumn vector
            double[][] deltaCurrent = calc.schurProduct(weightTimesDelta, colVector(calc.sigmoidPrimeList(weightedSumHidden)));
            // gradient of bias wrt cost function C, is also stored as a col vector (if i am not wrong ofc)
            // we need actviated list as a column vector also so, ignore
            // my bad programming skills, ill refine it some lazy day 
            double[][] weightUpdated = calc.dot(deltaCurrent, rowVector(inputLayer));
            nabla_weight[0] = weightUpdated;
            nabla_bias[0] = deltaCurrent;
        }
        // each day i look at this code and
        // realize whatever i did was wrong
        WeightBias gradient = new WeightBias();
        gradient.setBias(nabla_bias);
        gradient.setWeight(nabla_weight);
        return gradient;
    }
    Random random = new Random();

    /* If you have a dataset of 60,000 images, one epoch means the model has processed all 60,000 images once. 
    If you're training for 10 epochs, the model will process the entire dataset 10 times. n = no of training set and m = mini batch size */
    public void SDG(InputAndAcutalOutput trainingData, int epochs, int miniBatchSize, double eta) {
        double[][] inputs = trainingData.getInput();
        int[] labels = trainingData.getLabelOutput();
        int dataLength = trainingData.getDataLength();

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Create an index array for shuffling
            Integer[] indices = new Integer[dataLength];
            for (int i = 0; i < dataLength; i++) {
                indices[i] = i;
            }

            // we suffled the index, so that the order of data, and its corresponding actual label "y" is the same
            // yo milyena bhanye sab bhadragol huncha
            Collections.shuffle(Arrays.asList(indices), random);

            // suffle
            double[][] shuffledInputs = new double[dataLength][];
            int[] shuffledLabels = new int[dataLength];
            for (int i = 0; i < dataLength; i++) {
                shuffledInputs[i] = inputs[indices[i]];
                shuffledLabels[i] = labels[indices[i]];
            }

            // going through the suffled dataset mini batches
            for (int index = 0; index < dataLength; index += miniBatchSize) {
                int end = Math.min(index + miniBatchSize, dataLength);

                double[][] miniBatchInputs = Arrays.copyOfRange(shuffledInputs, index, end);
                int[] miniBatchLabels = Arrays.copyOfRange(shuffledLabels, index, end);
                updateMiniBatch(miniBatchInputs, miniBatchLabels, eta);
            }
            System.out.printf("Completed epoch: %d \n", (epoch + 1));
        }
    }

    private void print3DArray(double[][][] array3D) { // to debug vanishing gradient nvm
        for (int i = 0; i < array3D.length; i++) {
            System.out.println("Layer " + (i + 1) + ":");
            for (int j = 0; j < array3D[i].length; j++) {
                for (int k = 0; k < array3D[i][j].length; k++) {
                    System.out.print(array3D[i][j][k] + " ");
                }
                System.out.println();
            }
            System.out.println();
        }
    }

    private void print2DArray(double[][] array2D) {
        for (int i = 0; i < array2D.length; i++) {
            for (int j = 0; j < array2D[i].length; j++) {
                System.out.print(array2D[i][j] + " ");
            }
            System.out.println();
        }
    }

    // y = [1, 2, 3, 4 ...] list of actual labels in a suffled array corresponding to input x
    private void updateMiniBatch(double[][] inputLayer, int[] y, double eta) {
        double[][][] nabla_b = new double[biases.size()][][]; // size =  2
        double[][][] nabla_w = new double[weights.size()][][]; // size = 2

        for (int i = 0; i < nabla_b.length; i++) {
            nabla_b[i] = new double[biases.get(i).length][1];
        }
        for (int i = 0; i < nabla_w.length; i++) {
            nabla_w[i] = new double[weights.get(i).length][weights.get(i)[0].length];
        }
        // input.length = group of input data right,
        for (int digit = 0; digit < inputLayer.length; digit++) {
            double[] currentDigit = inputLayer[digit];
            double[] actualLabel = calc.oneHotEncode(this.outputSize, y[digit]);

            WeightBias slopeOfWeightBias = train(currentDigit, actualLabel);

            double[][][] slopeBias = slopeOfWeightBias.getBias();
            double[][][] slopeWeight = slopeOfWeightBias.getWeight();

            for (int i = 0; i < nabla_b.length; i++) {
                for (int j = 0; j < nabla_b[i].length; j++) {
                    nabla_b[i][j][0] += slopeBias[i][j][0];
                }
            }
            for (int i = 0; i < nabla_w.length; i++) {
                for (int j = 0; j < nabla_w[i].length; j++) {
                    for (int k = 0; k < nabla_w[i][j].length; k++) {
                        nabla_w[i][j][k] += slopeWeight[i][j][k];
                    }
                }
            }
        }

        // small things makes difference
        double lr = eta / inputLayer.length;
        for (int i = 0; i < weights.size(); i++) {
            for (int j = 0; j < weights.get(i).length; j++) {
                for (int k = 0; k < weights.get(i)[j].length; k++) {
                    weights.get(i)[j][k] -= lr * nabla_w[i][j][k];
                }
            }
        }
        for (int i = 0; i < biases.size(); i++) {
            for (int j = 0; j < biases.get(i).length; j++) {
                biases.get(i)[j][0] -= lr * nabla_b[i][j][0]; // since a col vector
            }
        }
    }

    public void test(String testSetPath, String testSetLabels) {
        System.out.println("testing....");
        double[][] inputLayer = loadImages(testSetPath); // 10,000 test images
        int[] labels = loadLabels(testSetLabels); // 10,000 images labels

        int correctPredictions = 0;
        int totalTest = labels.length;
        for (int i = 0; i < inputLayer.length; i++) {
            double[] currentInputLayer = inputLayer[i];
            double[] predicted = feedForward(currentInputLayer);

            int predictedLabel = argmax(predicted);

            if (predictedLabel == labels[i]) {
                correctPredictions++;
            }

        }
        double accuracy = (double) correctPredictions / totalTest * 100;
        long endTime = System.nanoTime();
        long duration = (endTime - startTime);  // nano second
        double durationInMinutes = (double) duration / 1_000_000_000 / 60;
        System.out.println("Execution time: " + durationInMinutes + " min");
        System.out.printf("Accuracy: %.2f%% (%d/%d) \n", accuracy, correctPredictions, totalTest);
    }

    private int argmax(double[] array) {
        int maxIndex = 0;
        double maxValue = array[0];

        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private double[][] loadImages(String datasetPath) {
        try (FileChannel fileChannel = new FileInputStream(datasetPath).getChannel()) {
            ByteBuffer buffer = ByteBuffer.allocateDirect(16); // header
            fileChannel.read(buffer);
            buffer.flip();

            // Read header information
            int magicNumber = buffer.getInt();
            if (magicNumber != 0x00000803) {
                throw new IllegalArgumentException("Invalid MNIST magic number: " + magicNumber);
            }
            System.out.println("Loading started: " + datasetPath);
            int numberOfImages = buffer.getInt();
            int rows = buffer.getInt();
            int cols = buffer.getInt();
            int area = rows * cols;
            ByteBuffer imageBuffer = ByteBuffer.allocateDirect(area);
            double[][] batchData = new double[numberOfImages][area];

            for (int batch = 0; batch < numberOfImages; batch++) {
                imageBuffer.clear();
                fileChannel.read(imageBuffer);
                imageBuffer.flip();

                for (int j = 0; j < area; j++) {
                    batchData[batch][j] = (imageBuffer.get() & 0xFF) / 255.0; // normalize binary value
                }
            }
            System.out.println("Loading completed.. :) ");
            return batchData;
        } catch (IOException ex) {
            throw new IllegalArgumentException("Error reading MNIST data: " + ex.getMessage());
        }
    }

    private int[] loadLabels(String trainingSetPath) {
        try (DataInputStream d = new DataInputStream(new FileInputStream(trainingSetPath))) {
            int magicNumber = d.readInt();
            int numberOfLabels = d.readInt();
            int[] labels = new int[numberOfLabels];
            for (int i = 0; i < numberOfLabels; i++) {
                labels[i] = d.readUnsignedByte();
            }
            return labels;
        } catch (IOException e) {
            throw new IllegalArgumentException(e.getMessage());
        }

    }

    // load mnist in batches and, then update weight and bias from that bais, then continue with another batches
    // else jvm cant handle all 60,000 data directly loaed into memory,
    // first (from constructor) wt and biases are gonna be randomly initlized and the updated
    public InputAndAcutalOutput loadMemory(String datasetPath, String trainingSetPath) {
        boolean dataset = false, trainingSet = false;
        double[][] images;
        int[] labelsImage;
        int imageLength;

        images = loadImages(datasetPath);
        labelsImage = loadLabels(trainingSetPath);

        if (images != null && labelsImage != null) {
            trainingSet = true;
            dataset = true;
        }
        imageLength = images.length;

        if (dataset && trainingSet) {
            InputAndAcutalOutput io = new InputAndAcutalOutput();
            io.setInput(images);
            io.setLabels(labelsImage);
            io.setImageLength(imageLength);
            return io;
        }
        throw new IllegalArgumentException("Unable to load data");
    }

}

class Main {

    public static void main(String[] args) {
        String trainData = "C:\\Users\\Aavash\\Desktop\\Java\\archive\\train-images-idx3-ubyte\\train-images-idx3-ubyte";
        String trainLabels = "C:\\Users\\Aavash\\Desktop\\Java\\archive\\train-labels-idx1-ubyte\\train-labels-idx1-ubyte";

        String testData = "C:\\Users\\Aavash\\Desktop\\Java\\archive\\t10k-images-idx3-ubyte\\t10k-images-idx3-ubyte";
        String testLabels = "C:\\Users\\Aavash\\Desktop\\Java\\archive\\t10k-labels-idx1-ubyte\\t10k-labels-idx1-ubyte";
        NeuralNetwork network = new NeuralNetwork(30); // my cpu will get fried if i increase more
        InputAndAcutalOutput trainingData = network.loadMemory(trainData, trainLabels);
        network.SDG(trainingData, 30, 10,  1);
        network.test(testData, testLabels);
    }
}
