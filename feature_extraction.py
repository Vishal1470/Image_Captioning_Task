import os
import pickle

def extract_features(image_dir, save_path):
    # Ensure parent folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = get_vgg_model()
    features = {}

    for img_name in os.listdir(image_dir):
        ...
    
    with open(save_path, "wb") as f:
        pickle.dump(features, f)

    return features
