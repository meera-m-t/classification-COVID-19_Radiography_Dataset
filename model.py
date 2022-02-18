from vgg16  import VGG
import tensorflow as tf 
import os
import skimage.io as io
import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping 


def read_imgs(path):
    covid_images = [io.imread(img) for img in glob.glob(f"{path}/COVID/*.png")] 
    covid_images = tf.convert_to_tensor(covid_images ,dtype=tf.int32)
    covid_labels = tf.fill([3616],1)
    train_covid_images, val_covid_images, test_covid_images = covid_images[:2170], covid_images[2170:2893], covid_images[2893:]
    train_covid_labels, val_covid_labels, test_covid_labels = covid_labels[:2170], covid_labels[2170:2893], covid_labels[2893:]
    print(train_covid_images.shape, val_covid_images.shape, test_covid_images.shape)
    

    opacity_images = [io.imread(img) for img in glob.glob(f"{path}/Lung_Opacity/*.png")] 
    opacity_images = tf.convert_to_tensor(opacity_images, dtype=tf.int32)
    opacity_labels = tf.fill([6012], 2)
    train_opacity_images, val_opacity_images, test_opacity_images = opacity_images[:3608], opacity_images[3608:4810], opacity_images[4810:]
    train_opacity_labels, val_opacity_labels, test_opacity_labels = opacity_labels[:3608], opacity_labels[3608:4810], opacity_labels[4810:]


    normal_images = [io.imread(img) for img in glob.glob(f"{path}/Normal/*.png")] 
    normal_images = tf.convert_to_tensor(normal_images, dtype=tf.int32)
    normal_labels = tf.fill([10192], 3)
    train_normal_images, val_normal_images, test_normal_images = normal_images[:6116], normal_images[6116:8154], normal_images[8154:]
    train_normal_labels, val_normal_labels, test_normal_labels = normal_labels[:6116], normal_labels[6116:8154], normal_labels[8154:]


    x_train = tf.ragged.constant([train_covid_images, train_opacity_images, train_normal_images], 0)
    y_train = tf.ragged.constant([train_covid_labels, train_opacity_labels, train_normal_labels], 0)

    x_val = tf.ragged.constant([val_covid_images, val_opacity_images, val_normal_images], 0)
    y_val = tf.ragged.constant([val_covid_labels, val_opacity_labels, val_normal_labels], 0)

    x_test = tf.ragged.constant([test_covid_images, test_opacity_images, test_normal_images], 0)
    y_test = tf.ragged.constant([test_covid_labels, test_opacity_labels, test_normal_labels], 0)    

    return x_train, y_train, x_val, y_val, x_test, y_test






if __name__ == '__main__':   
    path='/media/sameerahtalafha/easystore/project/COVID-19_Radiography_Dataset/preprocessed_data/'    
    length = 299  # Length of each Image
    width = 299  # Width of each Image
    model_name = 'VGG16_v2'  # DenseNet Models
    model_width = 16 # Width of the Initial Layer, subsequent layers start from here
    num_channel = 1  # Number of Input Channels in the Model
    problem_type = 'Regression' # Classification or Regression
    output_nums = 1  # Number of Class for Classification Problems, always '1' for Regression Problems
    x_train, y_train, x_val, y_val, x_test, y_test = read_imgs(path)
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)
    BATCH_SIZE = 64
    EPOCHS = 1
    print(x_train.shape, x_test.shape)
    Model = VGG(length, width, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, dropout_rate=False).VGG16_v2()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    callback = EarlyStopping(monitor='loss', patience=3)
    Model.summary()
    history = Model.fit(
        x_train,
        y_train,        
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[callback],
        validation_data=(x_val, y_val),
        shuffle=True
        )
    print(history.history.keys())
    Model.save("my_model")
    loss, acc = Model.evaluate(x_test, y_test, verbose = 0)
    print("Loss {}, Accuracy {}".format(loss, acc))

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('model_accuracy.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('model_loss.png') 
