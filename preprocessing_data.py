import glob
import os
from pathlib import Path

import cv2
import splitfolders


def preprocess_xray_dataset(path_dataset, path_workspace):
    names = [item for item in os.listdir(path_dataset) if os.path.isdir(os.path.join(path_dataset, item))]
    os.makedirs(f'{path_workspace}/preprocessed_data', exist_ok=True)
    for name in names:
        os.makedirs(f'{path_workspace}/preprocessed_data/{name}', exist_ok=True)
        for file in glob.glob(f"{path_dataset}/{name}/images/*.png"):
            name_file = os.path.basename(file)
            print(file, name_file)
            image = cv2.imread(file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(gray)

            cv2.imwrite(f"{path_workspace}/preprocessed_data/{name}/{name_file}", equalized)


def _preprocess_ct_scan_dataset(metadata_txt, classes, parent, imgs_loc):
    with metadata_txt.open("r") as train_images:
        os.makedirs(parent, exist_ok=True)
        for class_ in classes.values():
            os.makedirs(parent / class_, exist_ok=True)

        for image in train_images:
            [img_loc, img_class] = image.split()[0:2]
            img_file = imgs_loc / img_loc
            image = cv2.imread(str(img_file))
            img_class = classes[img_class]

            resized = cv2.resize(image, (299, 299))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(gray)
            cv2.imwrite(str(parent / img_class / img_file.name), equalized)
            print(f"Writing {str(parent / img_class / img_file.name)}")


def _assert_png_count(directory, number):
    count = len(glob.glob(f"{directory}/*.png"))
    assert count == number, f"Expected {number} images but found {count}"


def preprocess_ctscan_dataset(path_dataset, path_workspace):
    path_workspace = Path(path_workspace).resolve()
    path_dataset = Path(path_dataset).resolve()
    classes = {
        "0": "Normal",
        "1": "Viral Pneumonia",
        "2": "COVID"
    }

    TRAIN_TXT = "train_COVIDx_CT-2A.txt"
    VALID_TXT = "val_COVIDx_CT-2A.txt"
    TEST_TXT = "test_COVIDx_CT-2A.txt"
    IMG_DIR = "2A_images"

    train_loc = path_workspace / "Train"
    test_loc = path_workspace / "Test"
    valid_loc = path_workspace / "Valid"

    print("Writing Training Sets")
    _preprocess_ct_scan_dataset(path_dataset / TRAIN_TXT, classes, train_loc, path_dataset / IMG_DIR)
    _assert_png_count(train_loc / classes["0"], 35996)
    _assert_png_count(train_loc / classes["1"], 25496)
    _assert_png_count(train_loc / classes["2"], 82286)

    print("Writing Validation Sets")
    _preprocess_ct_scan_dataset(path_dataset / VALID_TXT, classes, valid_loc, path_dataset / IMG_DIR)
    _assert_png_count(valid_loc / classes["0"], 11842)
    _assert_png_count(valid_loc / classes["1"], 7400)
    _assert_png_count(valid_loc / classes["2"], 6244)

    print("Writing test tests")
    _preprocess_ct_scan_dataset(path_dataset / TEST_TXT, classes, test_loc, path_dataset / IMG_DIR)
    _assert_png_count(test_loc / classes["0"], 12245)
    _assert_png_count(test_loc / classes["1"], 7395)
    _assert_png_count(test_loc / classes["2"], 6018)


def split_data(path):
    splitfolders.ratio(f'{path}/preprocessed_data', output=f"{path}/SplitDataset", seed=1337, ratio=(.8, 0.2))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess the COVID-19_Radiography_Dataset/CT-SCAN Dataset')
    parser.add_argument("command", choices={"preprocess", "split"})
    parser.add_argument("dataset", choices={"X-Ray", "CT-Scan"})
    parser.add_argument(
        '-p', '--path_dataset', required=False, default='COVID-19_Radiography_Dataset', type=str,
        help='path of dataset')
    parser.add_argument(
        '-p_ws', '--path_workspace', required=False, default='.', type=str, help='path of dataset')

    args = parser.parse_args()

    if args.command == "preprocess":
        path_dataset = args.path_dataset
        path_workspace = args.path_workspace
        if args.dataset == "X-Ray":
            preprocess_xray_dataset(path_dataset, path_workspace)
        if args.dataset == "CT-Scan":
            preprocess_ctscan_dataset(path_dataset, path_workspace)
    else:
        split_data(args.path_workspace)


if __name__ == "__main__":
    main()
