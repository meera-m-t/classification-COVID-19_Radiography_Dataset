from vgg16  import VGG
import tensorflow as tf 
from SVDNet import SVDNet
from SVRNet import SVRNet
import skimage.io as io
from sklearn.metrics import confusion_matrix
import os
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.data.experimental import AUTOTUNE
import seaborn as sns
sns.set()

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
                    validation_split=None,
                    subset=None,
                    seed=1
                )
    return test_data.cache().prefetch(buffer_size=AUTOTUNE)         

if __name__ == '__main__':  
    path_train='output/train' 
    path_test='output/test' 

    length = 299  # Length of each Image
    width = 299  # Width of each Image
    # model_name = 'VGG16_v2'  # DenseNet Models
    model_name = 'SVRNet'   # DenseNet Models  
    # model_name = 'SVDNet'   # DenseNet Models
    model_width = 16 # Width of the Initial Layer, subsequent layers start from here
    num_channel = 1  # Number of Input Channels in the Model
    problem_type = 'Classification' # Classification or Regression
    output_nums = 2  # Number of Class for Classification Problems, always '1' for Regression Problems
 
    train_data, val_data = train_val_split(path_train, length)
    test_data = call_test_data(path_test, length)

    BATCH_SIZE = 32
    


    if os.path.isdir("SVRNet_model"): 
        # model = keras.models.load_model('vgg_model')
        # model = tf.keras.models.load_model('SVDNet_model')
        EPOCHS = 1
        model = tf.keras.models.load_model('SVRNet_model')
    else:
        EPOCHS = 100  
        #   model = VGG(length, width, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, dropout_rate=0.5).VGG16_v2()
    #     # # model = SVDNet(length, width, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, pooling='max', dropout_rate=0.5).SVDNet()  
        model = SVRNet(length, width, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, pooling='max', dropout_rate=0.5).SVRNet()  

    model.summary()

    dot_img_file = 'model_SVRNet.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
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

    if not os.path.isdir("SVRNet_model"):
        model.save("SVRNet_model")

    loss, acc = model.evaluate(test_data, verbose = 0, batch_size=32)
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
    plt.savefig('model_SVRNet_accuracy.png') 

    # # summarize history for loss
    plot2 = plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('model_SVRNet_lost.png')

    #confusion_matrix
    ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
            fmt='.2%', cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    fig = ax.get_figure()
    fig.savefig('model_SVRNet_confusion_matrix.png')

