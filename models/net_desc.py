import tensorflow as tf
from .net_utils import ConvModule, InceptionModule, DownsampleModule


class ModelSubClassing(tf.keras.Model):
    def __init__(self, num_classes):
        super(ModelSubClassing, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, strides=2, activation=tf.nn.relu)
        self.max1 = tf.keras.layers.MaxPooling2D(3)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.relu)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.drop = tf.keras.layers.Dropout(0.3)

        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.max1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.drop(x)
        x = self.gap(x)
        return self.dense1(x)


class MiniInception(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MiniInception, self).__init__()

        self.conv_block = ConvModule(96, (3, 3), (1, 1))

        self.inception_block1 = InceptionModule(32, 32)
        self.inception_block2 = InceptionModule(32, 48)
        self.downsample_block1 = DownsampleModule(80)

        self.inception_block3 = InceptionModule(112, 48)
        self.inception_block4 = InceptionModule(96, 64)
        self.inception_block5 = InceptionModule(80, 80)
        self.inception_block6 = InceptionModule(48, 96)
        self.downsample_block2 = DownsampleModule(96)

        self.inception_block7 = InceptionModule(176, 160)
        self.inception_block8 = InceptionModule(176, 160)

        self.avg_pool = tf.keras.layers.AveragePooling2D((7, 7))

        self.flat = tf.keras.layers.Flatten()
        self.classfier = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)

    def call(self, input_tensor, training=False, **kwargs):

        # forward pass
        x = self.conv_block(input_tensor)
        x = self.inception_block1(x)
        x = self.inception_block2(x)
        x = self.downsample_block1(x)

        x = self.inception_block3(x)
        x = self.inception_block4(x)
        x = self.inception_block5(x)
        x = self.inception_block6(x)
        x = self.downsample_block2(x)

        x = self.inception_block7(x)
        x = self.inception_block8(x)
        x = self.avg_pool(x)

        x = self.flat(x)
        return self.classfier(x)

    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
