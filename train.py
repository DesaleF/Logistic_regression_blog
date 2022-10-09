import pickle
import numpy as np
from datasets.datasets import load_dataset
from model.model import predict, initialize_params, optimize

def train(train_x, train_y, test_x, test_y, num_iterations, learning_rate, print_cost):

    w, b = initialize_params(train_x.shape[0])
    params, costs, grads = optimize(w,b, train_x, train_y, num_iterations, learning_rate, print_cost )
    w = params['w']
    b = params['b']
    dw = grads['dw']
    db = grads['db']

    # train and testset accuracty
    test_predictions = predict(w, b, test_x)
    print('Train Accuracy: {}'.format(100 - np.mean(np.abs(test_y - test_predictions))*100))
    checkpoint = {'losses': costs, 'w': w, 'b': b, 'learning_rate': learning_rate, 'num_iterations': num_iterations}
    return checkpoint

    checkpoint = model(train_x_standardized, train_y, test_x_standardized, test_y, num_iterations = 5000,learning_rate = 0.001, print_cost = True)
    
    
if __name__ == "__main__":
    train_x, train_y,test_x, test_y = load_dataset()
    num_px = train_x.shape[1]
    # to check the dimensions of each set
    print('The shape of train_x set is {}'.format(train_x.shape))
    print('The shape of train_y set is {}'.format(train_y.shape))
    print('The shape of test_x set is {}'.format(test_x.shape))
    print('The shape of test_y set is {}'.format(test_y.shape))
    
    train_x_flatten = train_x.reshape(train_x.shape[0],-1).T
    test_x_flatten = test_x.reshape(test_x.shape[0],-1).T
    print('Flatten train dataset dimension is now {}'.format(train_x.shape))
    print('Flatten test dataset dimension is now {}'.format(test_x.shape))
    train_x_standardized = train_x_flatten/255
    test_x_standardized = test_x_flatten/255

    checkpoint = train(
        train_x_standardized, 
        train_y, test_x_standardized, 
        test_y, num_iterations=5000,
        learning_rate=0.001, print_cost=True
    )

    # save weights and biases for inference
    with open('checkpoints/checkpoint.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)