using System;
using Numpy;
using Tensorflow;
//Keras is an api in tensorflow that makes it really easy to define neural networks
//Nerual network is a set of functions that can learn patterns
//Simplest possible network only has one neuron in it
//In keras, we use dense to define a layer of connected neurons
//Only 1 dense means only one layer
//You define the shape of your input into the network using the input_shape parameter, on the first layer
//The neural network has no relationship between x and y (training and test data) e.g y = x + 1...so given these
//values, it tries to guess the relationship between the two. It uses the previous guesses to see how good or bad its
//estimations are. The loss function measures this, then feeds that to the optimizer to improve/optimize on the next
//guess. Each guess should be better than the one before
//As the guesses get better, the accuracy gets closer to 100% and the term convergence is used

//Successive layers are defined in sequence, hence the keyword Sequential
//Neural networks deal with probability when they try to figure out answers to everything
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Utils;
//using Tensorflow.NumPy; // Python library that makes data representation in lists easier
using Tensorflow.Operations.Initializers;
using static Tensorflow.Binding;

namespace SensoryNeuralNetwork
{
    class Program
    {

        //private static Tensor sensor1Value = default!;
        //private static Tensor sensor2Value = default!;
        //private static Tensor sensor3Value = default!;

        private static Tensor sensor1Value = default!;
        private static Tensor sensor2Value = default!;
        private static Tensor sensor3Value = default!;


        private static double maxSensor1Value;
        private static double maxSensor2Value;
        private static double maxSensor3Value;

        static void Main(string[] args)
        {
            //Do something here

        }

        static void prepareModel()
        {
            // Preprocessing
            // Normalize the sensor data
            //Neural networks work better with normalized data
            var sensor1Data = Program.Sensor1Value / maxSensor1Value;
            var sensor2Data = Program.sensor2Value / maxSensor2Value;
            var sensor3Data = Program.sensor3Value / maxSensor3Value;

            // Feature engineering
            // Create new features by taking the log of sensor1Data and sensor2Data
            var logSensor1Data = Tensorflow.math_ops.log(sensor1Data);
            var logSensor2Data = Tensorflow.math_ops.log(sensor2Data);


            // Define the model architecture
            var layers = new LayersApi();


            // Concatenate the sensor data and the new features

            //Tensors are just buckets of numbers of a specific shape and a certain rank (dimensionality).
            //Tensors are used in Machine Learning with TensorFlow to represent input data and output data
            //(and everything in between) in Machine Learning models.
            //var inputData = new Tensor((NDarray)sensor1Data.size, (NDarray)5);
            var inputData = new Tensor((Tensorflow.NumPy.NDArray)sensor1Data.size, (Tensorflow.NumPy.NDArray)5);
            inputData = layers.Concatenate().Apply(new Tensors(sensor1Data, sensor2Data, sensor3Data, logSensor1Data, logSensor2Data));


            // input layer
            var inputs = layers.Input(5); // entry point into the network


            //Each layer of neurons needs an activation function that tells them what to do
            /**
             * Each layer of neurons need an activation function to tell them what to do. There are a lot of options, but just use these for now:

               ReLU effectively means:

                    if x > 0: 
                        return x

                    else: 
                        return 0

              In other words, it only passes values 0 or greater to the next layer in the network.


              Softmax takes a list of values and scales these so the sum of all elements will be equal to 1.
              When applied to model outputs, you can think of the scaled values as the probability for that class.
             * 
             */

            var x = layers.Dense(32, activation: "relu").Apply(inputs);
            x = layers.Dense(32, activation: "relu").Apply(x);
            x = layers.Dense(64, activation: "relu").Apply(x);
            x = layers.Dense(3, activation: "softmax").Apply(x);

            //output layer
            var outputs = layers.Dense(3).Apply(x);


            // Build the model
            //var model = new Sequential();

            var model = new Keras.Models.Model();
            //Apply(new Tensors(inputs, outputs));
            model.Summary();


            // Compile the model

            model.Compile(optimizer: "Adam", loss: "SparseCategoricalCrossentropy", new[] { "accuracy" });
            var outputData = new float[] { event1Data, event2Data, event3Data };

            //The training takes place in the fit command
            //We ask the model to figure out how to fit the input and output data
            //Epochs = 50 means it'll go through the training loop 50 times
            //Training loop = make a guess, measure how good or bad the guess is with the loss function, pass this data
            //to the optimizer, improve on the next guess
            var input_ndarray = np.array(new float[,] { { inputData.numpy() } });

            //model.Fit(inputData, outputData, epochs: 50, batch_size: 32, verbose: 1);
            model.Fit(input_ndarray, outputData, epochs: 50, batch_size: 32, verbose: 1);


            //Create sample test data to test our model
            var test_set = np.array(new float[,] { { 1, 2, 3, 4, 5 }, { 1, 3, 5, 7, 9 }, { 10, 20, 30, 40, 50 } });
            // var sensor2_test = np.array(new float[,] { { 1, 2, 3, 4, 5, 6 }, { 1, 3, 5, 7, 9, 11 } });
            //var sensor3_test = np.array(new float[,] { { 8, 7, 6, 5, 4, 3 }, { 0, 10, 8, 7, 12, 15 } });


            //var test_data = new Tensor((NDArray)sensor1_test.size, (NDArray)5);
            //test_data = layers.Concatenate().Apply(new Tensors(sensor1_test, sensor2_test, sensor3_test));

            //Let's test our model
            var test_ndarray = new Numpy.NDarray(test_set);

            var prediction = model.Predict(test_ndarray);
            Console.WriteLine(prediction);
        }

        public static Tensor Sensor1Value { get => sensor1Value; set => sensor1Value = value; }
        public static Tensor Sensor2Value { get => sensor2Value; set => sensor2Value = value; }
        public static Tensor Sensor3Value { get => sensor3Value; set => sensor3Value = value; }

        public static double MaxSensor1Value { get => maxSensor1Value; set => maxSensor1Value = value; }
        public static double MaxSensor2Value { get => maxSensor2Value; set => maxSensor2Value = value; }
        public static double MaxSensor3Value { get => maxSensor3Value; set => maxSensor3Value = value; }

        public static float event1Data { get; private set; }
        public static float event2Data { get; private set; }
        public static float event3Data { get; private set; }


    }
}



