from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    num_train = X.shape[0]
    for i in range(num_train):
        scores_raw = np.dot(X[i], W)
        scores_raw_moved = scores_raw - np.max(scores_raw, axis=0)
        probs = np.exp(scores_raw_moved)/np.sum(np.exp(scores_raw_moved), axis=0)
        loss += -np.log(probs[y[i]])
        dloss = probs
        dloss[y[i]] = dloss[y[i]] - 1
        dloss = dloss.reshape(-1,1)
        #print(dloss)
        Xi = X[i,:].reshape(1,-1)
        #print(dloss.shape, Xi.shape, np.dot(dloss, Xi).shape)
        dW += np.dot(dloss, Xi).T
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W
               

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    num_train = X.shape[0]
    
    scores_raw = np.dot(X, W) # (N, C)
    scores_raw_moved = scores_raw - np.max(scores_raw, axis=1).reshape(-1,1) # (N, C)
    probs = np.exp(scores_raw_moved)/np.sum(np.exp(scores_raw_moved), axis=1).reshape(-1,1) # (N, C)
    #print(probs[0,:], np.sum(probs[0,:]))
    loss += np.sum(-np.log(probs[range(num_train), y]))
    
    dloss = probs
    dloss[range(num_train), y] += - 1 # (N, C)

    dW += np.dot(dloss.T, X).T
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
