import string
import pickle

CAPTION_FILE = "../dataset/captions.txt"

def load_captions(file):
    mapping = {}
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            image, caption = line.strip().split(",", 1)
            image_id = image.split(".")[0]
            mapping.setdefault(image_id, []).append(caption)
    return mapping

def clean_captions(mapping):
    table = str.maketrans("", "", string.punctuation)
    for key, caps in mapping.items():
        for i in range(len(caps)):
            cap = caps[i].lower()
            cap = cap.translate(table)
            cap = "startseq " + cap + " endseq"
            caps[i] = cap

captions = load_captions(CAPTION_FILE)
clean_captions(captions)

with open("cleaned_captions.pkl", "wb") as f:
    pickle.dump(captions, f)

print("Captions cleaned and saved")
