# classification-COVID-19_Radiography_Dataset
This repository presents the implemntaion of SVRNet and SVDNet models in the paper [COVID-19 detection method based on SVRNet and SVDNet in lung x-rays](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8404611/pdf/JMI-008-017504.pdf).

![alt text](images/model.png)


## Data
COVID-19 Radiography Database (COVID-19 Chest X-ray Database)

You can download this dataset from [here](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database?fbclid=IwAR3JBdbiHVJFYHcNlR3r3Z1esKY3UKrCHJd8Nrhv4OPXdGhOZWtEcqtjEEg). This article selects 1560 lung x-rays from them as training set, validation set, and test set, including 1341 lung x-ray images that tested negative for COVID-19 and 219 lung x-ray images that tested positive for COVID-19. Before conducting the classification experiment on whether images contain COVID-19, the data set is divided into two categories. One type is used to train the model, including training set and validation set, which respectively account for 60% and 20% of the total data set; the other type is used to test the performance of the model, which accounts for 20% of the total data set. Image Preprocessing: OpenCV mainly relies on the cv2.imread() function to read the image, which can read the x-ray image from the original file. However, the contrast of the image at this time is unsatisfactory, which will affect the detection accuracy, so this article uses the limited contrast adaptive histogram equalization (CLAHE). Please run

```bash
$ python preprocessing_data.py
```

## Install
We used python 3.8 to run this code. TO install all requirements via pip:

```bash
$ pip install -r requirements.txt
```

## Model Training/Testing 
To Train the model, please choose which model you want to train in [main.py] 

```python
model = VGG(...)
model = SVRNet(...)
model = SVDNet(...)
```
Then run:

```bash
$ python main.py
```


