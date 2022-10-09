import pickle
import argparse
import numpy as np
from PIL import Image
from model.model import predict
from matplotlib.pyplot import imread


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='datasets/cat.jpeg', type=str)
    args = parser.parse_args()
    
    classes = {1:'cat', 0:'non cat'}
    image = np.array(imread(args.image_path))
    num_px = 64

    # We preprocess the image to fit your algorithm.
    my_image = np.array(Image.fromarray(image).resize(
        [num_px,num_px])).reshape((1, num_px*num_px*3)).T
    my_image = my_image/255
    
    # save weights and biases for inference
    with open('checkpoints/checkpoint.pkl', 'rb') as f:
        checkpoint = pickle.load(f)

    my_predicted_image = predict(checkpoint["w"], checkpoint["b"], my_image)
    print("y = " + str(np.squeeze(my_predicted_image)))
    print(f"predicted class: {classes[int(np.squeeze(my_predicted_image))]}.")
