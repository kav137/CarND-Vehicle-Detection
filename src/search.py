import numpy as np
import cv2
from src.utils import convert_color, draw_boxes
from src.features import get_hog_features
import matplotlib.pyplot as plt


class Search:
    search_regions = [
        (400, 475, 1.0),
        (410, 525, 1.0),
        (405, 525, 1.25),
        (405, 550, 1.5),
        (410, 550, 1.5),
        (410, 585, 1.75),
        (415, 585, 2),
        (425, 700, 2.85)
    ]

    @staticmethod
    def find_cars(img, ystart, ystop, scale, svc, orient, pix_per_cell, cell_per_block, cspace, draw_all_rect=False):
        img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]

        # apply color conversion if other than 'RGB'
        cropped_frame = img_tosearch
        cropped_frame = convert_color(img_tosearch, cspace)

        if scale != 1:
            imshape = cropped_frame.shape
            cropped_frame = cv2.resize(cropped_frame, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = cropped_frame[:, :, 0]
        ch2 = cropped_frame[:, :, 1]
        ch3 = cropped_frame[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) + 1

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        recs = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                test_prediction = svc.predict(hog_features.reshape(1, -1))

                if test_prediction == 1 or draw_all_rect == True:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    recs.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

        return recs

    @staticmethod
    def find_cars_with_search_regions(img, svc, orient, pix_per_cell, cell_per_block, cspace):
        boxes = []
        for search_region in Search.search_regions:
            boxes.append(
                Search.find_cars(
                    img,
                    search_region[0],
                    search_region[1],
                    search_region[2],
                    svc,
                    orient,
                    pix_per_cell,
                    cell_per_block,
                    cspace
                )
            )
        boxes = [item for sublist in boxes for item in sublist]
        return boxes

