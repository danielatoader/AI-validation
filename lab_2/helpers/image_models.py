# Contains wrappers for calling image models. Simply pass the image with the default values set.

from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import resnet
import tensorflow as tf
import PIL.Image
import os
import numpy as np
import random
import uuid
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
# from keras.utils.image_utils import load_img, img_to_array, array_to_img
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

# Tensorflow helper code to make sure we spread the load on the GPU if it's available
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    # Restrict TensorFlow to only allocate 8GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

# Tensorflow information for reference
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

class ImagePrediction():
    
    # Initiates our class and sets the model to VGG16
    def __init__(self, model_name="vgg16"):
        # Sets up our given model so we can run against different image recognition models
        # Default is VGG16
        match model_name:
            case "vgg16":
                self.model = vgg16.VGG16(weights='imagenet')
            case "vgg19":
                self.model = vgg19.VGG19(weights='imagenet')
            case "r50":
                self.model = resnet.ResNet50(weights='imagenet')
            case "r101":
                self.model = resnet.ResNet101(weights='imagenet')
            case "r152":
                self.model = resnet.ResNet152(weights='imagenet')
            case _:
                self.model = vgg16.VGG16(weights='imagenet')
    
    
    def convert_image_to_array(self, image):
        # Ensure the image is resized to the input specifications of 224x224
        original_image = load_img(image).resize([224, 224], PIL.Image.BILINEAR)
        # Image needs converting to an array for batch processing
        return img_to_array(original_image, dtype=int)
        
        
    def process_image(self, img_array):
        # Batch process the image so it can be passed to the model
        image_batch = np.expand_dims(img_array, axis=0)
        processed = preprocess_input(image_batch)
        return processed
    
    # Gets the top 10 predictions for the given image
    # param: image -> String - Location of the image to load
    # return: Array - Contains the image predictions
    def get_prediction(self, image):
        # Predict on the image
        predictions = self.model.predict(self.process_image(self.convert_image_to_array(image)), verbose=0)
        # Return predictions
        return decode_predictions(predictions, top=10)[0]
    
    
    # This method simply takes an image and randomly adds noise until the original prediction has changed
    def random_perturbations(self, image):
        # Convert the image to array so we can make changes to it later
        image_array = self.convert_image_to_array(image)
        # Process the image
        processed = self.process_image(image_array)
        # Get the initial predictions
        initial_prediction = current_prediction = decode_predictions(self.model.predict(processed, verbose=0), top=10)[0]
        
        # Index for loop
        index = 0
        # Whether we flipped the prediction
        flipped = False
        # While loop that breaks after so many iterations or once we flip the prediction
        while(initial_prediction[0][0] == current_prediction[0][0] and index <= 500):
            # Randomly mutate a pixel in our image array to black
            image_array[random.randrange(223)][random.randrange(223)] = [0] * 3
            # Process the new array and get a prediction from the model
            processed = self.process_image(image_array)
            current_prediction = decode_predictions(self.model.predict(processed, verbose=0), top=10)[0]
            # Print the details and update our variables depending on outcome of prediction
            print(f"current_prediction: {current_prediction[0][1]}, fitness: {current_prediction[0][2]}")
            if initial_prediction[0][0] != current_prediction[0][0]:
                flipped = True
            index += 1
        # Create a unique name for our output image
        save_name = f"output/{image.split('/')[1]}_{uuid.uuid4()}.jpg"
        # Convert the array we modified back to an image and save it
        image_to_save = array_to_img(image_array)
        image_to_save.save(save_name)
        # Print some details
        print(f"Image was successfully flipped from {initial_prediction[0][1]} to {current_prediction[0][1]} after ({index}) iterations!") if flipped else print(f"Image was not flipped in the given amount of iterations ({index})!")
        print(f"Modified image saved to: {save_name}")
        # Return the current prediction for further use if required
        return(current_prediction)