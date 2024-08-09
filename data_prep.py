from keras.preprocessing.image import ImageDataGenerator
from .config import train_dir, validation_dir, IMG_HEIGHT, IMG_WIDTH, batch_size

train_dir = train_dir
validation_dir = validation_dir

train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                            directory=train_dir,
                                                            shuffle=True,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            class_mode='categorical')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')
