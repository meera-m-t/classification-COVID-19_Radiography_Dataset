import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import splitfolders
import os

def preprocess(path_dataset, path_workspace):
    names = [item for item in os.listdir(path_dataset) if os.path.isdir(os.path.join(path_dataset, item))]
    for name in names:
        for file in glob.glob(f"{path}/{name}/*.png"):
            name_file = os.path.basename(file)
            image = cv2.imread(file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(gray)
            os.makedirs('{path_workspace}/preprocessed_data')
            cv2.imwrite(f"{path_workspace}/preprocessed_data/{name}/path_workspace {name_file", equalized)

def split_data(path):
    splitfolders.ratio(f'{path/preprocessed_data}', output="path/output", seed=1337, ratio=(.8, 0.2))

def main():
    parser = argparse.ArgumentParser(description='add, modify and delete upstream nodes')
    parser.add_argument(
       '-p','--path_dataset', required=False, default='COVID-19_Radiography_Dataset', type=str, help='path of dataset')
    parser.add_argument(
       '-p_ws','--path_workspace', required=False, default='classification-COVID-19_Radiography_Dataset', type=str, help='path of dataset')    
    args = parser.parse_args()
    path_dataset = args.path_dataset
    path_workspace = args.path_workspace
    preprocess(path_dataset ,path_workspace)
    split_data(path_workspace)

if __name__ == "__main__":
    main()
