import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt

# Path to the saved model
MODEL_PATH = 'working/my_cnn_model.h5'

# Load the saved model
model = tf.keras.models.load_model(MODEL_PATH)

# Classes
CLASSES = ['TRI5001', 'TRI5002', 'TRI5003','TRI5004']

def predict_emotion(img_path):
    # Load and preprocess the image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256,256), interpolation=cv2.INTER_LINEAR)
    img = np.array(image)
    img = img.reshape(1,256,256,1)
    
    # Predict the class
    predict_x = model.predict(img) 
    result = np.argmax(predict_x)
    
    # Print and display the result
    print("Predicted class:", CLASSES[result])
    plt.imshow(image, cmap='gray')
    plt.show()

# Example usage
# predict_emotion('data/train/TRI5003/IMG_2232.JPG')
predict_emotion('input/imagef.jpg')