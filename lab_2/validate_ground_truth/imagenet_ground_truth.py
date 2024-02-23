# Retrieves the ground truth of an Imagenet 2012 image
import scipy.io

class ImagenetGroundTruth():
    
    def __init__(self):
        # Loads the synsets from the meta.mat file
        self.synsets = scipy.io.loadmat('validate_ground_truth/meta_data/meta.mat')['synsets']
        # Creates a reference list for the ground truth index
        with open('validate_ground_truth/meta_data/ILSVRC2012_validation_ground_truth.txt') as f:
            self.truth_data = f.readlines()

    # Gets the ground truth index by referencing the name of the Imagenet image.
    # NOTE: Only valid for Imagenet images!
    def get_ground_truth(self, image_name):
        image = image_name.split('/')[1].split('_')[2].split('.')[0]
        ground_truth = self.truth_data[int(image) - 1]
        return(self.synsets[int(ground_truth) -1][0][1][0])

    # Validates the ground truth by comparing the given ground truth of an image.
    def validate_ground_truth(self, image_name, classification):
        return classification == self.get_ground_truth(image_name)