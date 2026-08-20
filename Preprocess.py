import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 128  # resize all images to 128x128

def load_data(dataset_path="dataset"):
    """
    Loads images from dataset/yes and dataset/no folders.
    Returns preprocessed images (X) and labels (y).
    """
    X = []
    y = []

    categories = {"yes": 1, "no": 0}  # yes = tumor, no = no tumor

    for category, label in categories.items():
        folder_path = os.path.join(dataset_path, category)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            # Read image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue  # skip unreadable files

            # Resize to fixed size
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Normalize pixel values (0-1)
            img = img / 255.0

            X.append(img)
            y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)

    return X, y


def split_data(X, y, test_size=0.2):
    """
    Splits data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X, y = load_data()
    print(f"Total images loaded: {len(X)}")
    print(f"Tumor images: {sum(y == 1)}")
    print(f"Non-tumor images: {sum(y == 0)}")
