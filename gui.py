import PySimpleGUI as sg
import os.path

# First the window layout in 2 columns
import tensorflow.keras as keras
import cv2
import numpy as np


svdnet_model_path = './SVDNet_model'
svdnet_model = keras.models.load_model(svdnet_model_path)

svrnet_model_path = './SVRNet_model'
svrnet_model = keras.models.load_model(svrnet_model_path)

vgg16_model_path = './VGG16_model'
vgg16_model = keras.models.load_model(vgg16_model_path)

classes = {
    0: "COVID",
    1: "Normal",
    2: "Viral Pneumonia"
}

MODELS = {
    "VGG": vgg16_model,
    "SVR": svrnet_model,
    "SVD": svdnet_model
}

def infer_for_filename(model, filename):
    image = cv2.imread(filename)
    numpy_arr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out = model.predict(np.expand_dims(numpy_arr, axis=0))
    prediction = np.argmax(out[0])
    return classes[prediction]


file_list_column = [
    [
        sg.Text("Image Classification USING Various Networks"),
        sg.Listbox(['VGG', 'SVR', 'SVD'], size=(10,3), enable_events=True, key='-MODEL-LIST-'),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(60, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Image Classifier", layout)


active_model_name = "VGG"

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif", ".jpg"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(filename=filename)
            class_name = infer_for_filename(MODELS[active_model_name], filename)
            window["-TOUT-"].update(os.path.basename(filename) +  f" {active_model_name} Model ===> " + class_name)
        except:
            pass

    elif event == "-MODEL-LIST-":
        try:
            active_model_name = values["-MODEL-LIST-"][0]
        except:
            pass

window.close()
