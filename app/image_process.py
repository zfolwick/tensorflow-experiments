import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import println as p
import pathlib
import matplotlib.pyplot as plt


logger = p.Logger
#%%

class ProcessFlowers:
    __data_dir = ""
    __purpose = ""
    def __init__(self):
        self.__purpose = "to process flowers!"
    
    def purpose(self):
        logger.log(self.__purpose)
    # get dataset
    def write_source_data(self, dataset_url):
        archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
        self.__data_dir = pathlib.Path(archive).with_suffix('')
        logger.log("wrote data to: " + str(self.__data_dir))

        # qa step: image count
        image_count = len(list(self.__data_dir.glob('*/*.jpg')))
        logger.log("completed downloading. Wrote " + str(image_count) + " to disk.")
        return self.__data_dir

    def displayFirstImage(self, images, i):
        #displays the first in a series of images
        roses = list(self.__data_dir.glob(images))
        img = PIL.Image.open(str(roses[i]))

    def load(self):
        batch_size = 32
        img_height = 180
        img_width = 180

        self.__training_dataset = tf.keras.utils.image_dataset_from_directory(
            self.__data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        self.__class_names = self.__training_dataset.class_names
        logger.log(str(self.__class_names))
        
        self.__validation_dataset = tf.keras.utils.image_dataset_from_directory(
            self.__data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
  
    def visualize(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self.__training_dataset.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                showImage = plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.__class_names[labels[i]])
                plt.axis("off")
            
        print("break point")
        
    def normalize(self):
        # rescale:
        normalization_layer = tf.keras.layers.Rescaling(1./255) # normalize
        normalized_ds = self.__training_dataset.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]
        # Notice the pixel values are now in `[0,1]`.
        print(np.min(first_image), np.max(first_image))
        
        
    def tune(self):
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = self.__training_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = self.__validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        
        num_classes = 5
        layer2 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        print(layer2)

        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255),
            layer2,
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes)
        ])
        
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])