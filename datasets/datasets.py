import h5py
import numpy as np
import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_datasets():
    """download trainset and testset from google drive
    """
    train_set = "1HF_bsnAiCbvwVl9f1joKJcgeZ5KT7ZE4"
    test_set = "1cHLNRprHJJTjUJhbch7qeAGKYdHlhoxC"
    
    download_file_from_google_drive(train_set, "datasets/train_catvnoncat.h5")
    download_file_from_google_drive(test_set, "datasets/test_catvnoncat.h5")


# Load the dataset that we will be working on
def load_dataset():
    """load training and testset from .h5 file

    Returns:
        array: train_x, train_y, test_x, test_y
    """
    # train_data
    with h5py.File('datasets/train_catvnoncat.h5', 'r') as train_data:
        train_x = np.array(train_data['train_set_x'][:])
        train_y = np.array(train_data['train_set_y'][:])
        # print(train_x.shape) # have good shape
        # print(train_y.shape) # shape is (209,) not good for vectorization
        train_y = train_y.reshape((1, train_y.shape[0]))
    # test data
    with h5py.File('datasets/test_catvnoncat.h5','r') as test_data:
        test_x = np.array(test_data['test_set_x'][:])
        test_y = np.array(test_data['test_set_y'][:])
        # reshape the test_y just like the train set
        test_y = test_y.reshape((1, test_y.shape[0]))
    return (train_x, train_y, test_x, test_y)


if __name__ == "__main__":
    download_datasets()
    