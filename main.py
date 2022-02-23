from vgg16  import VGG
import tensorflow as tf 
from SVDNet import SVDNet
import skimage.io as io
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.data.experimental import AUTOTUNE

def train_val_split(DATA_DIR,IMG_SIZE):

    train_data = image_dataset_from_directory(
                    DATA_DIR,
                    labels="inferred",
                    label_mode="categorical",
                    color_mode="grayscale",
                    batch_size=32,
                    image_size=(IMG_SIZE, IMG_SIZE),
                    validation_split=0.2,
                    subset="training",
                    seed=1
                )

    val_data = image_dataset_from_directory(
                    DATA_DIR,
                    labels="inferred",
                    label_mode="categorical",
                    color_mode="grayscale",
                    batch_size=32,
                    image_size=(IMG_SIZE, IMG_SIZE),
                    validation_split=0.2,
                    subset="validation",
                    seed=1
                )
    train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
    val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

    return train_data, val_data


def  call_test_data(DATA_DIR,IMG_SIZE):
    test_data = image_dataset_from_directory(
                    DATA_DIR,
                    labels="inferred",
                    label_mode="categorical",
                    color_mode="grayscale",   
                    batch_size=32,                 
                    image_size=(IMG_SIZE, IMG_SIZE),
                    validation_split=0.2,
                    subset="validation",
                    seed=1
                )
    return test_data          

if __name__ == '__main__':   
    path_train='output/train' 
    path_test='output/test'  

    length = 299  # Length of each Image
    width = 299  # Width of each Image
    # model_name = 'VGG16_v2'  # DenseNet Models
    model_name = 'SVDNet'   # DenseNet Models
    model_width = 3 # Width of the Initial Layer, subsequent layers start from here
    num_channel = 1  # Number of Input Channels in the Model
    problem_type = 'Classification' # Classification or Regression
    output_nums = 3  # Number of Class for Classification Problems, always '1' for Regression Problems
  
    train_data, val_data = train_val_split(path_train, length)
    test_data = call_test_data(path_test, length) 
    BATCH_SIZE = 32
    EPOCHS = 100
    
    # model = VGG(length, width, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, dropout_rate=0.5).VGG16_v2()
    model = SVDNet(length, width, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, pooling='max', dropout_rate=0.5).SVDNet()   
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()])

    callback = EarlyStopping(monitor='categorical_accuracy', patience=10)
    history = model.fit(
        train_data,    
        validation_data=val_data,  
        epochs=EPOCHS,
        callbacks=callback,
        batch_size=BATCH_SIZE,   
        shuffle=True
        )
    print(history.history.keys())
    model.save("my_model")
    loss, acc = model.evaluate(test_data, verbose = 0)
    print("Loss {}, Accuracy {}".format(loss, acc))

    # summarize history for accuracy
    plot1 = plt.figure(1)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('model_accuracy.png')  
    # # summarize history for loss

    plot2 = plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('model_lost.png')
