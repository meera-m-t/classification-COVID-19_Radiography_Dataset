import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage.io as io
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image_dataset_from_directory

from SVDNet_1 import SVDNet
from SVRNet import SVRNet
from vgg16 import VGG

sns.set()


def call_train_val_data(TRAIN_DIR, IMG_SIZE, VALID_DIR=None, BATCH_SIZE=32):
    if not VALID_DIR:
        train_data = image_dataset_from_directory(
                        TRAIN_DIR,
                        labels="inferred",
                        label_mode="categorical",
                        color_mode="grayscale",
                        batch_size=BATCH_SIZE,
                        image_size=(IMG_SIZE, IMG_SIZE),
                        validation_split=0.2,
                        subset="training",
                        seed=1
                    )

        val_data = image_dataset_from_directory(
                        TRAIN_DIR,
                        labels="inferred",
                        label_mode="categorical",
                        color_mode="grayscale",
                        batch_size=BATCH_SIZE,
                        image_size=(IMG_SIZE, IMG_SIZE),
                        validation_split=0.2,
                        subset="validation",
                        seed=1
                    )
    else:
        train_data = image_dataset_from_directory(
            TRAIN_DIR,
            labels="inferred",
            label_mode="categorical",
            color_mode="grayscale",
            batch_size=BATCH_SIZE,
            image_size=(IMG_SIZE, IMG_SIZE),
            subset=None,
            seed=1
        )

        val_data = image_dataset_from_directory(
            VALID_DIR,
            labels="inferred",
            label_mode="categorical",
            color_mode="grayscale",
            batch_size=BATCH_SIZE,
            image_size=(IMG_SIZE, IMG_SIZE),
            subset=None,
            seed=1
        )

    train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
    val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

    return train_data, val_data


def  call_test_data(DATA_DIR, IMG_SIZE):
    test_data = image_dataset_from_directory(
                    DATA_DIR,
                    labels="inferred",
                    label_mode="categorical",
                    color_mode="grayscale",   
                    batch_size=32,                                   
                    image_size=(IMG_SIZE, IMG_SIZE),
                    validation_split=None,
                    subset=None,
                    seed=1
                )
    return test_data.cache().prefetch(buffer_size=AUTOTUNE)         

def main():
    parser = argparse.ArgumentParser(description='determine the parameters in our model to classifiy x-ray images using three models (VGG16, SVRNet, SVDNet)')
    parser.add_argument(
       '-p_train','--path_train', required=False, default='output/train', type=str, help='path train')
    parser.add_argument(
        '-p_valid', '--path-valid', required=False, default=None, type=str, help="path valid"
    )
    parser.add_argument(
        '-p_test','--path_test', required=False, default='output/test', type=str, help='path test')
    parser.add_argument(
        '-l', '--length', required=False, default=299, type=int, help='length of image')
    parser.add_argument(
        '-w', '--width', required=False, default=299, type=int, help='width of image')
    parser.add_argument(
        '-m', '--model_name',choices=['VGG16', 'SVRNet', 'SVDNet'], required=False, default='VGG16', type=str, help='model_name')
    parser.add_argument(
        '-t', '--problem_type', choices=['Classification', 'Regression'], required=False, default='Classification', type=str, help='problem_type')
    parser.add_argument(
        '-o', '--output_nums', required=False, default=2, type=int, help='output_num')
    parser.add_argument(
        '-mw', '--model_width', required=False, default=16, type=int, help='width of the Initial Layer, subsequent layers start from here')
    parser.add_argument(
        '-c', '--num_channel', required=False, default=1, type=int, help='number of channel of the image')
    parser.add_argument(
          '-b', '--batch_size', required=False, default=32, type=int, help='batch size')
    parser.add_argument(
        '-e', '--epochs', required=False, default=15, type=int, help="Number of epochs"
    )

    parser.add_argument(
        '-p', '--prefix', required=False, default="chestxray", help="The prefix to add to all the saved files/models"
    )

    parser.add_argument(
        '-ct', '--conf-tick-prefix', required=False, default="COVID,Normal,Viral Pneumonia", help="Text Labels for the confusion matrix"
    )
  
    args = parser.parse_args()

    path_train = args.path_train
    path_valid = args.path_valid
    path_test = args.path_test
    length = args.length  
    width = args.width
    model_name = args.model_name
    model_width = args.model_width
    num_channel = args.num_channel
    problem_type = args.problem_type
    output_nums = args.output_nums
    prefix = args.prefix
    conf_tick_prefix = args.conf_tick_prefix.split(",")
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    train_data, val_data = call_train_val_data(path_train, length, path_valid, BATCH_SIZE)
    test_data = call_test_data(path_test, length)

    
    if model_name == 'VGG16':
        model = VGG(length, width, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, dropout_rate=0.5).VGG16_v2()
    elif model_name == 'SVRNet':
        model = SVRNet(length, width, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, pooling='max', dropout_rate=0.5).SVRNet()
    elif model_name == 'SVDNet':
        model = SVDNet(length, width, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, pooling='max', dropout_rate=0.3).SVDNet()
                    

    model.summary()

    dot_img_file = f'{model_name}_{prefix}.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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

    model.save(f'{model_name}_model_{prefix}')

    loss, acc = model.evaluate(test_data, verbose = 0)
    print("Loss {}, Accuracy {}".format(loss, acc))

    predictions = np.array([])
    labels =  np.array([])
    for x, y in test_data:
        predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=1)])
        labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

    cf_matrix = tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()


    # summarize history for accuracy
    plot1 = plt.figure(1)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'{model_name}_{prefix}_model_accuracy.png')

    # # summarize history for loss
    plot2 = plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'{model_name}_{prefix}_model_lost.png')

    #confusion_matrix
    plot3 = plt.figure(3)
    ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
            fmt='.2%', cmap='Blues')

    ax.set_title('Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical orderls
    ax.xaxis.set_ticklabels(conf_tick_prefix)
    ax.yaxis.set_ticklabels(conf_tick_prefix)
    fig = ax.get_figure()
    fig.savefig(f'{model_name}_{prefix}_model_confusion_matrix.png')


if __name__ == "__main__":
    main()



