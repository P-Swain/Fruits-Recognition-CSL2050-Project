import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from src.data_loader import load_images
from src.pca_transform import apply_incremental_pca
from src.ann_model import build_ann

# Load dataset
DATASET_PATH = "fruits-360_100x100/fruits-360/Training"
images, labels = load_images(DATASET_PATH)

# Encode labels
unique_labels, labels_encoded = np.unique(labels, return_inverse=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Apply Incremental PCA
X_train_pca, pca = apply_incremental_pca(X_train, n_components=150)
X_test_pca = pca.transform(X_test)

# Build and train ANN
model = build_ann(input_dim=X_train_pca.shape[1], num_classes=len(unique_labels))
model.fit(X_train_pca, y_train, epochs=10, batch_size=32, validation_data=(X_test_pca, y_test))

# Save model and PCA
model.save("fruit_ann_model.h5")
