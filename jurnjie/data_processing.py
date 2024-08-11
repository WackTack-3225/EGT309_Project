from keras.preprocessing.image import ImageDataGenerator
import time
import os

train_df = '/mnt/data/train'
test_df = '/mnt/data/test'
processed_dir = '/mnt/processed_data'


train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_df,
                                                    target_size = (64, 64),  # Resize 64x64
                                                    batch_size = 100,
                                                    shuffle = True,
                                                    class_mode = 'categorical',
                                                    save_to_dir=processed_dir)
                                                    

validation_generator = test_datagen.flow_from_directory(test_df,
                                                        target_size = (64, 64),  # Resize 64x64
                                                        batch_size = 100,
                                                        shuffle = True,
                                                        class_mode = 'categorical',
                                                            save_to_dir=processed_dir)
                                                            
     