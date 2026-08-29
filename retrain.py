import os
import cv2
import numpy as np
import tensorflow as tf
from core.model_manager import ModelManager

# This script finds collected custom data and fine-tunes the respective models.

DATASET_DIR = "C:/hand/custom_dataset"

def load_data_for_classes(class_list, img_size=(64, 64), mode="letter"):
    X = []
    y = []
    for i, cls in enumerate(class_list):
        folder_name = cls
        if cls == '*': folder_name = 'times'
        elif cls == '/': folder_name = 'div'
        elif cls == '<': folder_name = 'lt'
        elif cls == '>': folder_name = 'gt'
        elif cls == '?': folder_name = 'question'
        elif cls == '|': folder_name = 'pipe'
        
        d = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(d):
            continue
            
        for fname in os.listdir(d):
            if not fname.endswith('.png'): continue
            img_path = os.path.join(d, fname)
            roi = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if roi is None: continue
            
            # Preprocess the same way as model_manager
            h, w = roi.shape[:2]
            size = max(h, w)
            pad_h = (size - h) // 2
            pad_w = (size - w) // 2
            roi_square = cv2.copyMakeBorder(roi, pad_h, size - h - pad_h, pad_w, size - w - pad_w, cv2.BORDER_CONSTANT, value=0)
            
            if mode == "digit":
                roi_resized = cv2.resize(roi_square, (28, 28))
                normalized = roi_resized / 255.0
                X.append(normalized.reshape(28, 28, 1))
            else:
                roi_not = cv2.bitwise_not(roi_square)
                roi_resized = cv2.resize(roi_not, (64, 64))
                roi_bgr = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2BGR)
                normalized = roi_bgr / 255.0
                X.append(normalized.reshape(64, 64, 3))
            
            y.append(i)
            
    return np.array(X), np.array(y)

def retrain_model(model_path, X, y, num_classes):
    if len(X) == 0:
        return
    print(f"Retraining {model_path} with {len(X)} new samples...")
    
    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Recompile since it was loaded compile=False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    # Data Augmentation to simulate AR variability
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    
    model.fit(datagen.flow(X, y, batch_size=8), epochs=10, verbose=1)
    
    # Save back
    model.save(model_path)
    print(f"Model saved to {model_path}!")

def main():
    if not os.path.exists(DATASET_DIR):
        print("No custom dataset found at", DATASET_DIR)
        return
        
    mgr = ModelManager()
    
    # Train Letters
    X_let, y_let = load_data_for_classes(mgr.letter_classes, (64, 64), "letter")
    if len(X_let) > 0:
        retrain_model(mgr.letter_model_path, X_let, y_let, len(mgr.letter_classes))
        
    # Train Symbols
    X_sym, y_sym = load_data_for_classes(mgr.symbol_classes, (64, 64), "symbol")
    if len(X_sym) > 0:
        retrain_model(mgr.symbol_model_path, X_sym, y_sym, len(mgr.symbol_classes))
        
    # Train Digits
    digits = [str(i) for i in range(10)]
    X_dig, y_dig = load_data_for_classes(digits, (28, 28), "digit")
    if len(X_dig) > 0:
        retrain_model(mgr.digit_model_path, X_dig, y_dig, 10)
        
    print("Done retraining! Restart the AR app to use the updated models.")

if __name__ == "__main__":
    main()
