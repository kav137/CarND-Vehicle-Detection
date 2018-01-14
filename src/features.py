import numpy as np
import cv2
from skimage.feature import hog
from src.utils import convert_color
import matplotlib.image as mpimg

# just invoke native cv2 method with right parameters
def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True):
    # result could be either tuple or just a features array
    result = hog(
        img,
        orientations=orient,
        pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block),
        transform_sqrt=False,
        visualise=vis,
        feature_vector=feature_vec
    )
    return (result[0], result[1]) if vis else result

def extract_features(images_files, cspace='RGB', orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0, progress_title="Extract features"):
    features = []
    for file in images_files:
        # Read in each one by one
        image = mpimg.imread(file)
        feature_image = convert_color(image, cspace)

        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_feature = get_hog_features(
                    feature_image[:, :, channel],
                    orient,
                    pix_per_cell,
                    cell_per_block,
                    vis=False,
                    feature_vec=True
                )
                hog_features.append(hog_feature)
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(
                feature_image[:, :, hog_channel],
                orient,
                pix_per_cell,
                cell_per_block,
                vis=False,
                feature_vec=True
            )
        # Append the new feature vector to the features list

        features.append(hog_features)

    # Return list of feature vectors
    return features
