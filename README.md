# DL_assignment_1

**Libraries:**
  * Required libraries numpy, wandb, scikit-learn, matplotlib, and seaborn are imported.

**Dataset:**
  * The Fashion MNIST dataset has been downloaded.
  * 70,000 samples are split into training (90%) and testing (10%) sets. The training data is further split into training and validation data.
  * The split is done using the train_test_split library.

**Printing one image from each label:**
  * Using labels, one image from each label is selected.
  * Using matplotlib, 10 images are plotted in a grid (5x2).

**Activation Function:**
  * All required activation functions (sigmoid, tanh, relu, identity) are implemented along with their derivatives.
  * Activation functions take a pre-activation vector and apply transformations.

**Weights and Bias Initialization:**
  * Before starting the optimizer, weights and biases are initialized using two methods: 'random' and 'xavier'.
  * The input is reshaped from (28,28) to (784,1).

**Forward Propagation:**
  * It takes the weights, biases, and image. Using weights and biases, it finds the pre-activations and activations for each layer.
  * The input layer is the starting activation.

**Backward Propagation:**
  * It takes the weights (W), activations (H), pre-activation (A), and calculates the gradients of weights and biases.
  * It returns weight gradients (current_dw) and bias gradients (current_db).

**Train Accuracy:**
  * It takes the weights (W), biases (B), activation function, and loss function.
  * For every training image, it calls forward propagation to get the predicted label, compares it with the true label, and computes accuracy and loss.

**Validation Accuracy:**
  * It takes the weights (W), biases (B), activation function, and loss function.
  * For every validation image, it calls forward propagation to get the predicted label, compares it with the true label, and computes accuracy and loss.
  * I have written one function for calculating both train accuracy and validation accuracy. By calling it with data, it gives the accuracy and loss.

**Optimizations:**
  * Every optimization takes parameters such as layers, neurons, epochs, batch size, activation function, loss function, learning rate, and weight decay.
  * Each algorithm accumulates the summation of gradients of both weights (current_dw) and biases (current_db) returned by backpropagation and updates the weights 
    (W) and biases (B) for every batch of data points.

**Confusion Matrix:**
  * The best optimizer is tested on the test data and the results are plotted in a confusion matrix.

