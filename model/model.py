import numpy as np

#impostant functions

def sigmoid(z):
    # z = w.T X + b -> which is (1, m)
    s = 1/(1 + np.exp(-z)) # s is (1,m)
    
    return s


def initialize_params(dim):
    ''' Input:
            dim - is the number of input feature(#pixels)
        Outputs:
            w - weight vector with size (dim, 1)
            b - bias scalar(zero)
    '''
    # initialize weight with vector of small random numbers
    w = np.random.randn(dim,1)*0.001
    # initialize bias with zero
    b = 0
    return w, b
def forward_propagate(w, b, X):
    ''' Inputs:
            w - 12288x1 weight vector
            b - scalar
            X - 12288x209 numpy array-- 209 images of size 12288x1 stacked together in column 
        Output:
            A - 1x209 vector returned form sigmoid function
    '''
    # compute z
    z = np.dot(w.T, X) + b
    # compute activation using sigmoid function
    A = sigmoid(z)
    return A

def predict(w, b, X):
    ''' Inputs:
            w - 12288x1 weight vector(optimized)
            b - scalar (optimized)
            X - 12288xN numpy array-- N images of size 12288x1 stacked together in column 
        Output:
            predictions - 1xN vector containing of predicted labels for images in X
    '''
    # create vector same size as # of examples-(1, X.shape[1])
    predictions = np.zeros((1, X.shape[1]))
    A = forward_propagate(w, b, X)
    for i in range(A.shape[1]):
        if A[0,i] <= 0.5 :
            predictions[0,i] = 0
        else:
            predictions[0,i] = 1
    return predictions

def backward_propagation(X, A, Y):
    ''' Inputs:
            X - 12288x209 sized training set
            A - 1x209 vector of containing the output of forward pass
            Y - 1x209 vector having true labels of each images in X
        Outputs:
            cost - scalar value
            grads - computed gradients(dw, db) as dictionary
    '''
    m = X.shape[1]
    cost = (-1/m)* np.sum(np.dot(np.log(A),Y.T) + np.dot(np.log(1-A), (1-Y.T))) 
    # backward
    dz = (A-Y)
    dw = (1/m)*np.dot(X, dz.T)
    db = (1/m)*(np.sum(dz))
    grad = {'db':db,'dw': dw}
    cost = float(np.squeeze(cost)) 
    return cost, grad

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    ''' Inputs:
            w - a 12288x1 weight vector to be optimized
            b - a scalar bias value to be optimized
            X - 12288x209 training set
            Y - 1x209 true labels of 209 images in X
            num_iterations - is a scalar, # iteration for gradient descent to iterate
            learning_rate - scalar used to control the gradient descent update
            print_cost - boolean to decide whether to print or not cost during training
        Outputs:
            params - dictionary containing weights and bias
            loss - list of all costs computed in each iteration
            grads - dictionary of gradients computed using backpropagation
    
    '''
    costs = []
    
    for i in range(num_iterations):
        A = forward_propagate(w, b, X)
        cost,grads = backward_propagation(X, A, Y)
        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate*dw
        b = b - learning_rate*db

        # cost in each 100 iterations
        if i % 500 == 0:
            costs.append(cost)
            if print_cost:
                print('loss at itration {} is {}'.format(i, cost))
            # else:
                # print('.', end='')
                
        grads = {'dw': dw, 'db':db}
        params = {'w':w, 'b':b}
        
    return params, costs, grads

