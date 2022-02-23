import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import splitfolders
for name in ['COVID', 'Lung_Opacity', 'Normal']:
    for file in glob.glob(f"COVID-19_Radiography_Dataset/{name}/*.png"):
        name_file = os.path.basename(file)        
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)    
        cv2.imwrite(f"COVID-19_Radiography_Dataset/preprocessed_data/{name}/{name_file}", equalized)


splitfolders.ratio('COVID-19_Radiography_Dataset/preprocessed_data', output="output", seed=1337, ratio=(.8, 0.2))
