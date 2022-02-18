import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
for name in ['COVID', 'Lung_Opacity', 'Normal']:
    for file in glob.glob(f"/media/sameerahtalafha/easystore/project/COVID-19_Radiography_Dataset/{name}/*.png"):
        name_file = os.path.basename(file)        
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)    
        cv2.imwrite(f"/media/sameerahtalafha/easystore/project/COVID-19_Radiography_Dataset/preprocessed_data/{name}/{name_file}", equalized)

# img = cv2.imread("/media/sameerahtalafha/easystore/project/COVID-19_Radiography_Dataset/COVID/COVID-4.png", 0)

# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# img = clahe.apply(img) 
# # mask = cv2.threshold(img,200,255,cv2.THRESH_BINARY)[1]
# # dst = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
# # cv2.imwrite('clahe_2.jpg',dst)


# mask = np.zeros(img.shape[:2], dtype="uint8")
# masked = cv2.bitwise_and(img, img, mask=mask)
# img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
# cv2.imwrite('clahe_2.jpg',img)

