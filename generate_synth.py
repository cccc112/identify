import cv2
import numpy as np
import os
import random

symbols = ['+', '-', '*', '/', '=', '(', ')', '<', '>']
DATASET_DIR = 'C:/hand/custom_dataset'

for sym in symbols:
    folder_name = sym
    if sym == '*': folder_name = 'times'
    elif sym == '/': folder_name = 'div'
    elif sym == '<': folder_name = 'lt'
    elif sym == '>': folder_name = 'gt'
    
    d = os.path.join(DATASET_DIR, folder_name)
    os.makedirs(d, exist_ok=True)
    
    for i in range(100):
        img = np.zeros((100, 100), dtype=np.uint8)
        font_scale = random.uniform(1.5, 3.5)
        thickness = random.randint(3, 7)
        font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_PLAIN])
        
        # Random offsets
        ox = random.randint(-15, 15)
        oy = random.randint(-15, 15)
        
        cv2.putText(img, sym, (25 + ox, 65 + oy), font, font_scale, 255, thickness, cv2.LINE_AA)
        
        # Add some noise
        noise = np.random.randint(0, 50, (100, 100), dtype=np.uint8)
        img = cv2.add(img, noise)
        
        cv2.imwrite(os.path.join(d, f'synth_{i}.png'), img)

print('Synthetic data generated!')
