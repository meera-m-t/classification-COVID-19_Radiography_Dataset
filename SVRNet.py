import tensorflow as tf
from  tensorflow.keras.utils import to_categorical 

def Conv_2D_Block(inputs, model_width, kernel, strides):
    # 2D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv2D(model_width, kernel, strides=strides, padding="same", kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x



def stem(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    conv = Conv_2D_Block(inputs, num_filters, (3, 3), (1,1))
    conv = Conv_2D_Block(conv, num_filters, (3, 3), (1,1))
    if conv.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="same")(conv)
    else:
        pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv)    
    return pool


def dense_block_1(x, num_filters):  
    conv = Conv_2D_Block(x, num_filters, (3, 3), (1,1))
    conv = Conv_2D_Block(conv, num_filters, (3, 3), (1,1))
    x = tf.keras.layers.concatenate([x, conv], axis=-1)
    if x.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="same")(x)
    else:
        pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)


    return pool



def dense_block_2(x, num_filters):
    conv = Conv_2D_Block(x, num_filters, (3, 3), (1,1))
    conv = Conv_2D_Block(conv, num_filters, (3, 3), (1,1))
    x = tf.keras.layers.concatenate([x, conv], axis=-1)
    if x.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="same")(x)
    else:
        pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    return pool



def dense_block_3(x, num_filters):
    conv = Conv_2D_Block(x, num_filters, (3, 3), (1,1))
    conv = Conv_2D_Block(conv, num_filters, (3, 3), (1,1))
    conv = Conv_2D_Block(conv, num_filters, (1, 1), (1,1))    
    x = tf.keras.layers.concatenate([x, conv], axis=-1)
    if x.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="same")(x)
    else:
        pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    return pool


def dense_block_4(x, num_filters):
    conv = Conv_2D_Block(x, num_filters, (3, 3), (1,1))
    conv = Conv_2D_Block(conv, num_filters, (3, 3), (1,1))
    conv = Conv_2D_Block(conv, num_filters, (1, 1), (1,1))    
    x = tf.keras.layers.concatenate([x, conv], axis=-1)
    return x


def classifier(inputs, class_number):
    # Construct the Classifier Group
    # inputs       : input vector
    # class_number : number of output classes
    out = tf.keras.layers.Dense(class_number, activation='softmax')(inputs)

    return out


def regressor(inputs, feature_number):
    # Construct the Regressor Group
    # inputs         : input vector
    # feature_number : number of output features
    out = tf.keras.layers.Dense(feature_number, activation='linear')(inputs)

    return out


class SVRNet:
    def __init__(self, length, width, num_channel, num_filters, problem_type='Classification',
                 output_nums=1, pooling='max', dropout_rate=False):
        # length: Input Signal Length
        # model_depth: Depth of the Model
        # model_width: Width of the Model
        # kernel_size: Kernel or Filter Size of the Input Convolutional Layer
        # num_channel: Number of Channels of the Input Predictor Signals
        # problem_type: Regression or Classification
        # output_nums: Number of Output Classes in Classification mode and output features in Regression mode
        # pooling: Choose either 'max' for MaxPooling or 'avg' for Averagepooling
        # dropout_rate: If turned on, some layers will be dropped out randomly based on the selected proportion
    
        self.length = length
        self.width = width
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        
    def MLP(self, x):
        outputs = []
        if self.pooling == 'avg':                     
            x = tf.keras.layers.GlobalAveragePooling2D()(x)

        elif self.pooling == 'max':            
            x = tf.keras.layers.GlobalMaxPooling2D()(x)

        # Final Dense Outputting Layer for the outputs
        x = tf.keras.layers.Flatten(name='flatten')(x)   
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)    

        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        # Problem Types
        if self.problem_type == 'Classification':
            # The Classifier for n classes
            class_number = self.output_nums
            outputs = classifier(x, class_number)
        elif self.problem_type == 'Regression':
            # The Regressor [Default]
            feature_number = self.output_nums
            outputs = regressor(x, feature_number)

        return outputs

    def SVRNet(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)
        stem_block = stem(x, self.num_filters)  # The Stem Convolution Group   

        Dense_Block_1 = dense_block_1(stem_block, self.num_filters * 2)
        Dense_Block_2 = dense_block_2(Dense_Block_1, self.num_filters * 4)
        Dense_Block_3 = dense_block_3(Dense_Block_2, self.num_filters * 8)
        Dense_Block_4 = dense_block_4(Dense_Block_3, self.num_filters * 8)        
        outputs = self.MLP(Dense_Block_4)
        model = tf.keras.Model(inputs, outputs)

        return model