#Step 1: Install Required Libraries and Import Data
#First, let's install the necessary libraries and download the LFW dataset. You can directly download it from Kaggle or use sklearn.datasets.fetch_lfw_people for easy access.


# Install required libraries
!pip install scikit-learn
!pip install opencv-python

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import exposure
import cv2

#Step 2: Download and Preprocess the LFW Dataset
#Let's fetch the dataset and preprocess it.


# Fetch the LFW dataset (Labelled Faces in the Wild)
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Get the images and labels
X = lfw_people.images  # Image data
y = lfw_people.target  # Labels
target_names = lfw_people.target_names  # Person names

# Flatten the images for PCA or use them as raw pixels for CNN
n_samples, h, w = X.shape
X_flattened = X.reshape((n_samples, h * w))

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.3, random_state=42)

#Step 3: Eigenfaces (PCA) for Feature Extraction

# Apply PCA (Eigenfaces) for dimensionality reduction
n_components = 150  # You can adjust this
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

# Project the images onto the Eigenface space
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Train an SVM classifier using the PCA features
svm_clf = SVC(kernel='rbf', class_weight='balanced')
svm_clf.fit(X_train_pca, y_train)

# Evaluate the classifier
accuracy = svm_clf.score(X_test_pca, y_test)
print(f"SVM with PCA accuracy: {accuracy:.2f}")

#Step 4: HOG (Histogram of Oriented Gradients) Feature Extraction

# Extract HOG features from the images
def extract_hog_features(images):
    hog_features = []
    for img in images:
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_features.append(fd)
    return np.array(hog_features)

# Extract HOG features for train and test sets
X_train_hog = extract_hog_features(X_train.reshape(n_samples, h, w))
X_test_hog = extract_hog_features(X_test.reshape(n_samples, h, w))

# Train an SVM classifier using HOG features
svm_clf_hog = SVC(kernel='rbf', class_weight='balanced')
svm_clf_hog.fit(X_train_hog, y_train)

# Evaluate the classifier
accuracy_hog = svm_clf_hog.score(X_test_hog, y_test)
print(f"SVM with HOG accuracy: {accuracy_hog:.2f}")

#Step 5: LBP (Local Binary Patterns) Feature Extraction

# Extract LBP features from the images
def extract_lbp_features(images):
    lbp_features = []
    for img in images:
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        lbp_features.append(np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))[0])
    return np.array(lbp_features)

# Extract LBP features for train and test sets
X_train_lbp = extract_lbp_features(X_train.reshape(n_samples, h, w))
X_test_lbp = extract_lbp_features(X_test.reshape(n_samples, h, w))

# Train an SVM classifier using LBP features
svm_clf_lbp = SVC(kernel='rbf', class_weight='balanced')
svm_clf_lbp.fit(X_train_lbp, y_train)

# Evaluate the classifier
accuracy_lbp = svm_clf_lbp.score(X_test_lbp, y_test)
print(f"SVM with LBP accuracy: {accuracy_lbp:.2f}")

#Step 6: CNN Model for Facial Recognition

from tensorflow.keras import layers, models

# Build CNN model
cnn_model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(h, w, 1)),  # Normalize the image data
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(target_names), activation='softmax')  # Output layer for classification
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN
cnn_model.fit(X_train.reshape(-1, h, w, 1), y_train, epochs=10, batch_size=32, validation_data=(X_test.reshape(-1, h, w, 1), y_test))

# Evaluate the CNN model
cnn_accuracy = cnn_model.evaluate(X_test.reshape(-1, h, w, 1), y_test)
print(f"CNN accuracy: {cnn_accuracy[1]:.2f}")
