import tensorflow as tf


class ConvModule(tf.keras.layers.Layer):
    def __init__(self, kernel_num, kernel_size, strides, padding="same"):
        super(ConvModule, self).__init__()

        self.conv = tf.keras.layers.Conv2D(
            kernel_num, kernel_size=kernel_size, strides=strides, padding=padding
        )

        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)


class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, kernel_size1x1, kernel_size3x3):
        super(InceptionModule, self).__init__()

        self.conv1 = ConvModule(kernel_size1x1, kernel_size=(1, 1), strides=(1, 1))
        self.conv2 = ConvModule(kernel_size3x3, kernel_size=(3, 3), strides=(1, 1))
        self.cat = tf.keras.layers.Concatenate()

    def call(self, inputs, training=False):
        x_1x1 = self.conv1(inputs)
        x_3x3 = self.conv2(inputs)
        x = self.cat([x_1x1, x_3x3])
        return x


class DownsampleModule(tf.keras.layers.Layer):
    def __init__(self, kernel_size):
        super(DownsampleModule, self).__init__()

        self.conv3 = ConvModule(
            kernel_size, kernel_size=(3, 3), strides=(2, 2), padding="valid"
        )

        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
        self.cat = tf.keras.layers.Concatenate()

    def call(self, input_tensor, training=False):
        conv_x = self.conv3(input_tensor, training=training)
        pool_x = self.pool(input_tensor)

        return self.cat([conv_x, pool_x])
