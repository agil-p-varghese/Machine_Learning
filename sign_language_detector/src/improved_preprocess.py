import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    labels = df['label'].values
    images = df.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
    return images, labels

def augment_data(X, y):
    """Apply data augmentation to increase dataset diversity"""
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    return datagen

def get_enhanced_data(train_csv):
    X, y = load_data(train_csv)
    
    # Add noise and variations to simulate real-world conditions
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X)):
        img = X[i]
        label = y[i]
        
        # Original image
        X_augmented.append(img)
        y_augmented.append(label)
        
        # Add gaussian noise
        noise = np.random.normal(0, 0.05, img.shape)
        noisy_img = np.clip(img + noise, 0, 1)
        X_augmented.append(noisy_img)
        y_augmented.append(label)
        
        # Add brightness variation
        bright_img = np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)
        X_augmented.append(bright_img)
        y_augmented.append(label)
    
    X_aug = np.array(X_augmented)
    y_aug = np.array(y_augmented)
    
    y_cat = to_categorical(y_aug, num_classes=26)
    return train_test_split(X_aug, y_cat, test_size=0.1, random_state=42)

def preprocess_live_image(hand_img):
    """Enhanced preprocessing for live prediction"""
    if hand_img.size == 0:
        return None
    
    # Convert to grayscale
    if len(hand_img.shape) == 3:
        gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = hand_img
    
    # Apply histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Resize to model input size
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize
    normalized = resized.astype(np.float32) / 255.0
    
    # Reshape for model input
    return normalized.reshape(1, 28, 28, 1)

def get_test_data(test_csv):
    X_test, y_test = load_data(test_csv)
    y_test_cat = to_categorical(y_test, num_classes=26)
    return X_test, y_test_cat