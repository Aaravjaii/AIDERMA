

import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
def load_dataset(dataset_path):
    """
    Load images from the Kaggle dataset
    Expected folder structure:
    dataset_path/
    ├── Oily/
    ├── Dry/
    └── Normal/
    """
    print("Loading dataset...")
    
    features = []  
    labels = []    
    
    skin_types = ['Oily', 'Dry', 'Normal']
    
    for skin_type in skin_types:
        folder_path = os.path.join(dataset_path, skin_type)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found!")
            continue
            
        print(f"Processing {skin_type} images...")
        
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            
            
            image = cv2.imread(img_path)
            if image is not None:
                
                image_features = extract_features_for_training(image)
                
                features.append(image_features)
                labels.append(skin_type)
        
        print(f"Processed {len([f for f in image_files if cv2.imread(os.path.join(folder_path, f)) is not None])} {skin_type} images")
    
    return np.array(features), np.array(labels)

def extract_features_for_training(image):
    """Extract the same 7 features your original code uses"""
    

    img = cv2.resize(image, (500, 500))
    blur = cv2.GaussianBlur(img, (5, 5), 1)
    

    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    mean_L = np.mean(L)
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    
    
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    pore_percentage = np.sum(cleaned == 255) / cleaned.size * 100
    
    
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    sebum_mask = cv2.inRange(hsv, (0, 0, 200), (30, 80, 255))
    sebum_percentage = np.sum(sebum_mask == 255) / sebum_mask.size * 100
    
    
    edges = cv2.Canny(gray, 50, 150)
    wrinkle_factor = np.sum(edges > 0) / edges.size * 100
    
    
    return [mean_L, mean_a, mean_b, texture_score, pore_percentage, sebum_percentage, wrinkle_factor]


def train_model(features, labels):
    """Train a Random Forest model on the extracted features"""
    
    print("Training machine learning model...")
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    

    model = RandomForestClassifier(
        n_estimators=100,  
        random_state=42,
        max_depth=10
    )
    
    model.fit(X_train_scaled, y_train)
    
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.2%}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    
    feature_names = ['Brightness', 'Redness', 'Yellowness', 'Texture', 'Pores', 'Sebum', 'Wrinkles']
    importances = model.feature_importances_
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.title('Feature Importance in Skin Type Classification')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    return model, scaler


def save_model(model, scaler, model_path='skin_model.pkl', scaler_path='scaler.pkl'):
    """Save the trained model and scaler for later use"""
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model saved as {model_path}")
    print(f"Scaler saved as {scaler_path}")


def load_trained_model(model_path='skin_model.pkl', scaler_path='scaler.pkl'):
    """Load the previously trained model and scaler"""
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler


def predict_with_trained_model(features, model, scaler):
    """Use the trained model to predict skin type"""
    
    
    features_scaled = scaler.transform([features])
    
    
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    
    max_prob = np.max(probability)
    
    return prediction, max_prob


def main_training():
    """Main function to run the complete training process"""
    
    
    dataset_path = r"C:\Users\arwac\Downloads\AI DERMA\AI DERMA\archive\Oily-Dry-Skin-Types\train"
 
    
    
    features, labels = load_dataset(dataset_path)
    
    if len(features) == 0:
        print("No data loaded! Check your dataset path.")
        return
    
    print(f"Loaded {len(features)} images total")
    print(f"Unique skin types: {np.unique(labels)}")
    
    
    model, scaler = train_model(features, labels)
    
    
    save_model(model, scaler)
    
    print("\nTraining complete! Your model is ready to use.")


if __name__ == "__main__":
    main_training()