import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

IMAGE_DIR = "../dataset/images"

vgg = VGG16()
vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)

features = {}

for img_name in tqdm(os.listdir(IMAGE_DIR)):
    path = os.path.join(IMAGE_DIR, img_name)
    image = load_img(path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, *image.shape))
    image = preprocess_input(image)

    feature = vgg.predict(image, verbose=0)
    image_id = img_name.split(".")[0]
    features[image_id] = feature

with open("image_features.pkl", "wb") as f:
    pickle.dump(features, f)

print("Image features extracted and saved")
