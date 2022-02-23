import tensorflow as tf


def Conv_2D_Block(inputs, model_width, kernel):
    # 2D Convolutional Block with BatchNormalization
    conv = tf.keras.layers.Conv2D(model_width, (kernel, kernel), padding='same')(inputs)
    batch_norm = tf.keras.layers.BatchNormalization()(conv)
    activate = tf.keras.layers.Activation('relu')(batch_norm)

    return activate

class VGG:
    def __init__(self, length, width, num_channel, num_filters, problem_type='Classification',
                 output_nums=3, dropout_rate=True):
        self.length = length
        self.width = width
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.dropout_rate = dropout_rate

    def VGG16_v2(self):
            inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
            x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)
            # Block 1
            x = Conv_2D_Block(x, self.num_filters * (2 ** 0), 3)
            x = Conv_2D_Block(x, self.num_filters * (2 ** 0), 3)
            if x.shape[1] <= 2:
                x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
            else:
                x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

            # Block 2
            x = Conv_2D_Block(x, self.num_filters * (2 ** 1), 3)
            x = Conv_2D_Block(x, self.num_filters * (2 ** 1), 3)
            if x.shape[1] <= 2:
                x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
            else:
                x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

            # Block 3
            x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 3)
            x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 3)
            x = Conv_2D_Block(x, self.num_filters * (2 ** 2), 1)
            if x.shape[1] <= 2:
                x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
            else:
                x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

            # Block 4
            x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
            x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
            x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 1)
            if x.shape[1] <= 2:
                x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
            else:
                x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

            # Block 5
            x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
            x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 3)
            x = Conv_2D_Block(x, self.num_filters * (2 ** 3), 1)
            if x.shape[1] <= 2:
                x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2), padding="valid")(x)
            else:
                x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

            # Fully Connected (MLP) block
            x = tf.keras.layers.Flatten(name='flatten')(x)
            x = tf.keras.layers.Dense(4096, activation='relu')(x)
            x = tf.keras.layers.Dense(4096, activation='relu')(x)
            if self.dropout_rate:
                x = tf.keras.layers.Dropout(self.dropout_rate, name='Dropout')(x)
            if self.problem_type == 'Regression':    
                outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
            elif self.problem_type == 'Classification':
                outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

            # Create model.
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            return model

