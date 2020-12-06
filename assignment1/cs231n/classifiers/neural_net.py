from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        
        l1_output = (np.dot(X, self.params['W1']) + self.params['b1']).clip(min=0) # (N, hidden_size)
        scores = np.dot(l1_output, self.params['W2']) + self.params['b2'] # (N, output_size)
        
        w1_out = np.dot(X, self.params['W1'])
        l1_lin = w1_out + self.params['b1']
        l1_relu = np.maximum(l1_lin, 0)
        w2_out = np.dot(l1_relu, self.params['W2'])
        l2_lin = w2_out + self.params['b2']
        scores = l2_lin.copy()

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        def softmax_forward(x):
            x = x - np.max(x, axis=1).reshape(-1,1) # (N, C)
            exps = np.exp(x)
            
            sexps = np.sum(exps, axis=1).reshape(-1,1)
            divexps = (1/sexps).reshape(-1,1)
            mul = exps * divexps
            return mul, {"exps":exps, "sexps":sexps, "divexps":divexps}

        def softmax_backward(dout, cache):
            softmax_grad = {}

            dexps_0 = cache["divexps"] * dout
            softmax_grad["dexps_0"] = dexps_0

            ddivexps = np.sum(cache["exps"] * dout, axis=1).reshape(-1,1)
            softmax_grad["ddivexps"] = ddivexps

            dsexps = -1.0/(cache["sexps"]**2) * ddivexps
            softmax_grad["dsexps"] = dsexps

            # dexps_1 = dsexps/(np.sum(cache["exps"], axis=1).reshape(-1,1))*cache["exps"]
            dexps_1 = dsexps * np.ones(dout.shape)
            softmax_grad["dexps_1"] = dexps_1

            dexps = dexps_0 + dexps_1
            softmax_grad["dexps"] = dexps

            dx = cache["exps"] * (dexps)
            softmax_grad["dx"] = dx

            return softmax_grad

        softmax, cache = softmax_forward(l2_lin)
        select_pyi = softmax[range(X.shape[0]), y].reshape(-1,1)
        #print(select_pyi)
        neg_log = -np.log(select_pyi)
        loss_data = np.mean(neg_log)
        w1_l2 = np.sum(self.params['W1'] * self.params['W1'])
        w2_l2 = np.sum(self.params['W2'] * self.params['W2'])
        b1_l2 = np.sum(self.params['b1'] * self.params['b1'])
        b2_l2 = np.sum(self.params['b2'] * self.params['b2'])
        loss_reg = (w1_l2 + w2_l2 + b1_l2 + b2_l2) * reg
        loss = loss_data + loss_reg

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
    
        dout = 1
        dloss_data = 1
        dneg_log = np.ones(neg_log.shape)/neg_log.shape[0]
        #print(dneg_log) #0.2
        dselect_pyi = -1/select_pyi * dneg_log
        #print(select_pyi)
        #print(dselect_pyi)
        dsoftmax_0 = np.zeros(softmax.shape)
        np.add.at(dsoftmax_0,tuple([range(X.shape[0]), y]),1)
        dsoftmax = dsoftmax_0 * dselect_pyi
        #print(dsoftmax)
        dl2_lin = softmax_backward(dsoftmax, cache)
        grads['b2'] = np.sum(dl2_lin["dx"], axis=0) + 2 * reg * self.params['b2']
        grads['W2'] = np.dot(l1_relu.T, dl2_lin["dx"]) + 2 * reg * self.params['W2']
        dl1_relu = np.dot(dl2_lin["dx"], self.params['W2'].T)
        dl1_lin = np.maximum(l1_lin, 0)
        dl1_lin[dl1_lin > 0] = 1
        dl1_lin *= dl1_relu
        grads['b1'] = np.sum(dl1_lin, axis=0) + 2 * reg * self.params['b1']
        grads['W1'] = np.dot(X.T, dl1_lin) + 2 * reg * self.params['W1']
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            idx = np.random.randint(X.shape[0], size=batch_size)
            X_batch = X[idx,:]
            y_batch = y[idx]
            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass
            self.params['b2'] -= grads["b2"] * learning_rate
            self.params['W2'] -= grads["W2"] * learning_rate
            self.params['b1'] -= grads["b1"] * learning_rate
            self.params['W1'] -= grads["W1"] * learning_rate

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        l1_output = (np.dot(X, self.params['W1']) + self.params['b1']).clip(min=0) # (N, hidden_size)
        scores = np.dot(l1_output, self.params['W2']) + self.params['b2'] # (N, output_size)
        
        w1_out = np.dot(X, self.params['W1'])
        l1_lin = w1_out + self.params['b1']
        l1_relu = np.maximum(l1_lin, 0)
        w2_out = np.dot(l1_relu, self.params['W2'])
        l2_lin = w2_out + self.params['b2']

        y_pred = np.argmax(l2_lin, axis=1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
