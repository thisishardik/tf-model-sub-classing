import tensorflow as tf
import config
from models import callbacks
from models.net_desc import ModelSubClassing, MiniInception
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# print(tf.config.list_physical_devices("GPU"))

raw_input = config.INPUT_SHAPE

model = MiniInception(num_classes=config.NUM_OF_CLASSES)
model.build((None, *raw_input))
model.build_graph(raw_input).summary()

# tf.keras.utils.plot_model(
#     model.build_graph(raw_input),
#     to_file="model.png",
#     dpi=96,
#     show_shapes=True,
#     show_layer_names=True,
#     expand_nested=False,
# )

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = x_train.astype("float32") / 255

x_test = x_test.astype("float32") / 255

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

y_train = tf.keras.utils.to_categorical(
    y_train, num_classes=config.NUM_OF_CLASSES)
y_test = tf.keras.utils.to_categorical(
    y_test, num_classes=config.NUM_OF_CLASSES)


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(
    buffer_size=1024).batch(config.BATCH_SIZE)


val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(config.BATCH_SIZE)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.AUC()],
    loss_weights=None,
    sample_weight_mode=None,
    weighted_metrics=None,
)

tensorBoardCallback = tf.keras.callbacks.TensorBoard(
    log_dir=f"{os.path.join(config.CWD, '/logs')}",
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    write_steps_per_second=True,
    update_freq="epoch",
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None,
)

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=None,
    epochs=config.EPOCHS,
    verbose=1,
    validation_data=val_dataset,
    steps_per_epoch=config.STEPS_PER_EPOCH,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    callbacks=[tensorBoardCallback],
)

# Look from custom training loop and launching tensorboard
# Ref: https://towardsdatascience.com/model-sub-classing-and-custom-training-loop-from-scratch-in-tensorflow-2-cc1d4f10fb4e
