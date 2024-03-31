import numpy as np
from pathlib import Path

def load_data_small():
    """ Load small training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    script_dir = Path(__file__).parent
    train_all = np.loadtxt(f'{script_dir}/data/smallTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt(f'{script_dir}/data/smallValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_medium():
    """ Load medium training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    script_dir = Path(__file__).parent
    train_all = np.loadtxt(f'{script_dir}/data/mediumTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt(f'{script_dir}/data/mediumValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_large():
    """ Load large training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    script_dir = Path(__file__).parent
    train_all = np.loadtxt(f'{script_dir}/data/largeTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt(f'{script_dir}/data/largeValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def linearForward(input, p):
    """
    :param input: input vector (column vector) WITH bias feature added
    :param p: parameter matrix (alpha/beta) WITH bias parameter added
    :return: output vector
    """
    return np.dot(p,input)


def sigmoidForward(a):
    """
    :param a: input vector WITH bias feature added
    """
    return 1/(1+np.exp(-a))


def softmaxForward(b):
    """
    :param b: input vector WITH bias feature added
    """
    numerator=np.exp(b)
    denominator=np.sum(np.exp(b))
    return numerator/denominator


def crossEntropyForward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    :return: float
    """
    return float(-(1/hot_y.shape[1])*np.sum(hot_y*np.log(y_hat)))

def one_hot_encoder(y,num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[y]=1
    return one_hot

def NNForward(x, y, alpha, beta):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: alpha WITH bias parameter added
    :return: all intermediate quantities x, a, z, b, y, J #refer to writeup for details
    TIP: Check on your dimensions. Did you make sure all bias features are added?
    """
    #  Convert y to one-hot encoding
    # a = # Apply linear transformation
    # z = # Apply sigmoid activation

    # Add bias term to hidden layer output before passing to output layer
    # z_with_bias

    # b = # Forward Pass through output layer using linearForward with augmented z
    # y_hat = # Apply softmax to get probabilities

    # Compute the cross-entropy loss
    # J = 
    
    # return x, a, z_with_bias, b, y_hat, J
    a=linearForward(x,alpha)
    z=np.insert(sigmoidForward(a),0,1).reshape(-1,1)
    b=linearForward(z,beta)
    y1=softmaxForward(b)
    y=one_hot_encoder(y,len(y1)).reshape(-1,1)
    J=crossEntropyForward(y,y1)
    return x,a,z,b,y1,J


def softmaxBackward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    """
    return y_hat-hot_y



def linearBackward(prev, p, grad_curr):
    """
    :param prev: previous layer WITH bias feature
    :param p: parameter matrix (alpha/beta) WITH bias parameter
    :param grad_curr: gradients for current layer
    :return:
        - grad_param: gradients for parameter matrix (alpha/beta)
        - grad_prevl: gradients for previous layer
    TIP: Check your dimensions.
    """
    grad_param = np.dot(grad_curr, prev.reshape(-1,1).T)
    grad_prevl = np.dot(p.T, grad_curr)
    return grad_param, grad_prevl


def sigmoidBackward(curr, grad_curr):
    """
    :param curr: current layer WITH bias feature
    :param grad_curr: gradients for current layer
    :return: grad_prevl: gradients for previous layer
    TIP: Check your dimensions
    """
    sigmoid_derivative = curr * (1 - curr)

    # Compute gradients for the previous layer
    grad_prev = grad_curr * sigmoid_derivative

    return grad_prev


def NNBackward(x, y, alpha, beta, z, y_hat):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: alpha WITH bias parameter added
    :param z: z as per writeup
    :param y_hat: vector of probabilistic distribution for predicted label
    :return:
        - grad_alpha: gradients for alpha
        - grad_beta: gradients for beta
        - g_b: gradients for layer b (softmaxBackward)
        - g_z: gradients for layer z (linearBackward)
        - g_a: gradients for layer a (sigmoidBackward)
    """
    # Convert y to one-hot encoding
    y_one_hot =one_hot_encoder(y,10).reshape(-1,1)
    
    # Gradient of Cross Entropy Loss w.r.t. y_hat
    g_y_hat =softmaxBackward(y_one_hot,y_hat)
    
    # Gradient of Loss w.r.t. beta (Weights from hidden to output layer)
    grad_beta, g_b = linearBackward(z,beta,g_y_hat)
    
    # Gradient of Loss w.r.t. activation before sigmoid (a)
    g_a =sigmoidBackward(z,g_b)
    
    g_b=g_b[1:,:]
    g_a=g_a[1:,:]
    # Gradient of Loss w.r.t. alpha (Weights from input to hidden layer)
    grad_alpha, g_x =linearBackward(x,alpha,g_a)
    
    # return grad_alpha, grad_beta, g_y_hat, g_b_no_bias, g_a
    return grad_alpha,grad_beta,g_y_hat,g_b,g_a



def SGD(tr_x, tr_y, valid_x, valid_y, hidden_units, num_epoch, init_flag, learning_rate):
    """
    :param tr_x: Training data input (size N_train x M)
    :param tr_y: Training labels (size N_train x 1)
    :param tst_x: Validation data input (size N_valid x M)
    :param tst_y: Validation labels (size N_valid x 1)
    :param hidden_units: Number of hidden units
    :param num_epoch: Number of epochs
    :param init_flag:
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
    :param learning_rate: Learning rate
    :return:
        - alpha weights
        - beta weights
        - train_entropy (length num_epochs): mean cross-entropy loss for training data for each epoch
        - valid_entropy (length num_epochs): mean cross-entropy loss for validation data for each epoch
    """
    m=tr_x.shape[1]
    d=hidden_units
    # Initialize weights
    if init_flag==False:
        alpha=np.zeros((d,m+1))
        beta=np.zeros((10,d+1))
    else:
        alpha=np.random.uniform(-0.1,0.1,size=(d,m+1))
        beta=np.random.uniform(-0.1,0.1,size=(10,d+1))
        alpha[:, 0] = 0
        beta[:,0]=0
    train_entropy=[]
    test_entropy=[]
    # Itarate over epochs
    for e in range(num_epoch):
        # Itarate over training data
        for i in range(tr_x.shape[0]):
            X,Y=tr_x[i],tr_y[i]
            # Forward pass for a single training sample
            x,a,z,b,y1,J=NNForward(np.insert(X,0,1),Y,alpha,beta)
            # Backward pass for a single training sample
            grad_alpha, grad_beta, g_y_hat, g_b_no_bias, g_a=NNBackward(x,Y,alpha,beta,z,y1)
            # Update weights
            alpha-=learning_rate*grad_alpha
            beta-=learning_rate*grad_beta
         # Calculate mean training loss for the epoch
        train_entropy.append(np.mean([NNForward(np.insert(tr_x[p], 0, 1), tr_y[p], alpha, beta)[-1] for p in range(len(tr_x))]))
         # Validation forward pass
        test_entropy.append(np.mean([NNForward(np.insert(valid_x[p], 0, 1), valid_y[p], alpha, beta)[-1] for p in range(len(valid_y))]))
    return alpha,beta,train_entropy,test_entropy



def prediction(tr_x, tr_y, valid_x, valid_y, tr_alpha, tr_beta):
    """
    :param tr_x: Training data input (size N_train x M)
    :param tr_y: Training labels (size N_train x 1)
    :param valid_x: Validation data input (size N_valid x M)
    :param valid_y: Validation labels (size N-valid x 1)
    :param tr_alpha: Alpha weights WITH bias
    :param tr_beta: Beta weights WITH bias
    :return:
        - train_error: training error rate (float)
        - valid_error: validation error rate (float)
        - y_hat_train: predicted labels for training data
        - y_hat_valid: predicted labels for validation data
    """
    train_error,valid_error=0,0
    y_hat_train,y_hat_valid=[],[]
    for i in range(tr_x.shape[0]):
        x,y=tr_x[i],tr_y[i]
        x,a,z,b,y_hat,J=NNForward(np.insert(x,0,1),y,tr_alpha,tr_beta)
        l=max(y_hat)
        y1=np.where(y_hat==l)[0][0]
        if y1!=y1:
            train_error+=1
        y_hat_train.append(y1)
    train_error/=tr_x.shape[0]
    for i in range(valid_x.shape[0]):
        x,y=valid_x[i],valid_y[i]
        x,a,z,b,y_hat,J=NNForward(np.insert(x,0,1),y,tr_alpha,tr_beta)
        l=max(y_hat)
        y1=np.where(y_hat==l)[0][0]
        if y1!=y1:
            valid_error+=1
        y_hat_valid.append(y1)
    
    valid_error/=valid_x.shape[0]
    return train_error,valid_error,y_hat_train,y_hat_valid

### FEEL FREE TO WRITE ANY HELPER FUNCTIONS

def train_and_valid(X_train, y_train, X_val, y_val, num_epoch, num_hidden, init_rand, learning_rate):
    """ Main function to train and validate your neural network implementation.

        X_train: Training input in N_train-x-M numpy nd array. Each value is binary, in {0,1}.
        y_train: Training labels in N_train-x-1 numpy nd array. Each value is in {0,1,...,K-1},
            where K is the number of classes.
        X_val: Validation input in N_val-x-M numpy nd array. Each value is binary, in {0,1}.
        y_val: Validation labels in N_val-x-1 numpy nd array. Each value is in {0,1,...,K-1},
            where K is the number of classes.
        num_epoch: Positive integer representing the number of epochs to train (i.e. number of
            loops through the training data).
        num_hidden: Positive integer representing the number of hidden units.
        init_flag: Boolean value of True/False
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
        learning_rate: Float value specifying the learning rate for SGD.

        RETURNS: a tuple of the following six objects, in order:
        loss_per_epoch_train (length num_epochs): A list of float values containing the mean cross entropy on training data after each SGD epoch
        loss_per_epoch_val (length num_epochs): A list of float values containing the mean cross entropy on validation data after each SGD epoch
        err_train: Float value containing the training error after training (equivalent to 1.0 - accuracy rate)
        err_val: Float value containing the validation error after training (equivalent to 1.0 - accuracy rate)
        y_hat_train: A list of integers representing the predicted labels for training data
        y_hat_val: A list of integers representing the predicted labels for validation data
    """
    ### YOUR CODE HERE
    loss_per_epoch_train = []
    loss_per_epoch_val = []
    err_train = None
    err_val = None
    y_hat_train = None
    y_hat_val = None

    alpha,beta,loss_per_epoch_train,loss_per_epoch_val=SGD(X_train, y_train, X_val, y_val, num_hidden, num_epoch, init_rand, learning_rate)

    err_train,err_val,y_hat_train,y_hat_val=prediction(X_train, y_train, X_val, y_val, alpha,beta)
    
    return (loss_per_epoch_train, loss_per_epoch_val,
            err_train, err_val, y_hat_train, y_hat_val)