import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    # BEGIN YOUR CODE
    return x + y
    # END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    # BEGIN YOUR CODE

    # read images
    with gzip.open(image_filename, "rb") as image_file:
        header_bytes = image_file.read(16)
        header, num_images, num_rows, num_columns = struct.unpack(
            '>Iiii', header_bytes)
        if header != 0x00000803:
            raise Exception("invalid header for image file")
        image_size = num_rows * num_columns
        dataset_size = num_images * image_size
        image_data = np.frombuffer(
            image_file.read(dataset_size), dtype=np.uint8)
        # normalize
        min = np.min(image_data)
        max = np.max(image_data)
        image_data = (image_data - min) / (max - min)
        # shape
        images = np.reshape(image_data.astype(
            np.float32), (num_images, image_size))

    # r read labels
    with gzip.open(label_filename, "rb") as label_file:
        header_bytes = label_file.read(8)
        header, num_labels = struct.unpack('>Ii', header_bytes)
        if header != 0x00000801:
            raise Exception("invalid header for label file")
        labels = np.frombuffer(label_file.read(num_labels), dtype=np.uint8)

    return images, labels

    # END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    # BEGIN YOUR CODE

    # Verbose version
    # loss = []
    # for zi in range(Z.shape[0]):
    #     l = np.log(np.sum(np.exp(Z[zi])))
    #     l -= Z[zi, y[zi]]
    #     loss.append(l)
    # softmax = np.average(loss)

    # Semi verbose
    # zy = Z[np.arange(Z.shape[0]), y]
    # zlog = np.log(np.sum(np.exp(Z), axis=1))
    # zdiff = zlog - zy
    # softmax = np.average(zdiff)

    # One liner
    softmax = np.average(np.log(np.sum(np.exp(Z), axis=1)
                                ) - Z[np.arange(Z.shape[0]), y])

    return softmax

    # END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr=0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    # BEGIN YOUR CODE

    num_classes = theta.shape[1]
    num_examples = X.shape[0]

    for i in range(0, num_examples, batch):
        # Batch
        x_batch = X[i:i+batch]
        y_batch = y[i:i+batch]
        batch_size = y_batch.shape[0]

        # One hot encoding (batch_size x num_classes)
        I = np.zeros((batch_size, num_classes))
        I[np.arange(batch_size), y_batch] = 1

        # Softmax probabilities (batch_size x num_classes)
        Z = x_batch @ theta
        Z_exp = np.exp(Z)
        Z_sum = np.reshape(np.sum(Z_exp, axis=1), (Z_exp.shape[0], 1))
        Z_norm = Z_exp / Z_sum

        # Gradient given the input batch, one-hot labels and softmax
        gradient = x_batch.T @ (Z_norm - I)

        # Apply learning rate to the batch
        theta -= (lr / batch) * gradient

    # END YOUR CODE


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    # BEGIN YOUR CODE

    # Gradient is derivative of L cross-entropy (Sigma(XW1)W2, y) with respect to W1 and W2.

    num_examples = X.shape[0]
    num_classes = W2.shape[1]

    for i in range(0, num_examples, batch):
        # Batch
        X_batch = X[i:i+batch]
        Y_batch = y[i:i+batch]
        batch_size = Y_batch.shape[0]

        # One hot encoding (batch_size x num_classes)
        I = np.zeros((batch_size, num_classes))
        I[np.arange(batch_size), Y_batch] = 1

        # Z
        Z = X_batch @ W1
        Z[Z < 0] = 0  # ReLu

        # G2
        G2 = np.exp(Z @ W2)
        G2_sum = np.reshape(np.sum(G2, axis=1), (-1, 1))
        G2 = (G2 / G2_sum) - I  # normalize

        # G1
        G1 = G2 @ W2.T
        Z1 = np.copy(Z)
        Z1[Z1 != 0] = 1  # 1 when positive, 0 otherwise
        G1 = np.multiply(Z1, G1)

        # Gradients
        W1_gradient = (X_batch.T @ G1) / batch
        W2_gradient = (Z.T @ G2) / batch

        W1 -= (lr) * W1_gradient
        W2 -= (lr) * W2_gradient

    # END YOUR CODE


# CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h, y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h, y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(
        np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1, 0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1, 0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"
              .format(epoch, train_loss, train_err, test_loss, test_err))


if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr=0.2)
