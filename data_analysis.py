import os
from PIL import Image
import matplotlib.pyplot as plt

IMAGE_DIR = "../dataset/images"
CAPTION_FILE = "../dataset/captions.txt"

images = os.listdir(IMAGE_DIR)
print("Total Images:", len(images))

with open(CAPTION_FILE, "r", encoding="utf-8") as f:
    captions = f.readlines()

print("Total Captions:", len(captions))

print("\nSample Captions:")
for i in range(5):
    print(captions[i].strip())

img = Image.open(os.path.join(IMAGE_DIR, images[0]))
plt.imshow(img)
plt.axis("off")
plt.show()
