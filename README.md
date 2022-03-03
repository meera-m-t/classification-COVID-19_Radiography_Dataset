# COVID-19 detection method based on SVRNet and SVDNet in lung x-rays
This repository presents the implementation of `SVRNet` and `SVDNet` models from the paper [COVID-19 detection method based on SVRNet and SVDNet in lung x-rays](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8404611/pdf/JMI-008-017504.pdf).
![model](images/model.png)
## Install
We used python 3.8 to run this code. To install all requirements via pip:
```bash
$ pip install -r requirements.txt
```
## Data: COVID-19 Radiography Database (COVID-19 Chest X-ray Database)
You can download this dataset from [here](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database?fbclid=IwAR3JBdbiHVJFYHcNlR3r3Z1esKY3UKrCHJd8Nrhv4OPXdGhOZWtEcqtjEEg). The 11045 lung x-rays (Normal/COVID) images are selected as training set, and 2781 images (Normal/COVID) as test set. To improve contrast in the images, the limited contrast adaptive histogram equalization (CLAHE) is applied. Run:
```bash
$ python preprocessing_data.py
```
## Train the model
To Train the model, Run: 
```bash
python --model_name ['VGG16', 'SVRNet', 'SVDNet']  main.py
```
