import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# there is no need to use ugly if statements
def convert_color(image, cspace):
    if cspace == 'RGB':
        return np.copy(image)

    color_name = 'COLOR_RGB2' + cspace
    target_color = getattr(cv2, color_name)

    return cv2.cvtColor(image, target_color)

def draw_boxes(img, boxes, color=(0, 0, 255)):
    draw_img = np.copy(img)
    for box in boxes:
        cv2.rectangle(draw_img, box[0], box[1], color, 2)
    return draw_img

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap  # Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()

        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 3)

    return img


def visualize_input_data(files, title, cols=10, rows=3, random=False):
    fig = plt.figure(figsize=(15, 5))
    for row in range(rows):
        for col in range(cols):
            axis = fig.add_subplot(
                rows,
                cols,
                row * cols + col + 1,
                xticks=[],
                yticks=[]
            )
            image_index = np.random.randint(low=0, high=len(files)) if random else row * cols + col
            image = mpimg.imread(files[image_index])
            axis.imshow(image)
    plt.suptitle(title, fontsize=24)
    plt.show()


# def show_pair(image1, image2, title):
#     fig = plt.figure(figsize=(10, 4))

#     axis1 = fig.add_subplot(121, xticks=[], yticks=[])
#     axis1.imshow(image1)
#     axis2 = fig.add_subplot(122, xticks=[], yticks=[])
#     axis2.imshow(image2)

#     if not (title is None):
#         plt.suptitle(title, fontsize=18)

#     plt.show()

def show_rows(rows):
    rows_total = len(rows)
    cols_total = len(rows[0])

    fig = plt.figure(figsize=(12, 3*rows_total - 1))
    for row in range(rows_total):
        for col in range(cols_total):
            axis = fig.add_subplot(
                rows_total,
                cols_total,
                row * cols_total + col + 1,
                xticks=[],
                yticks=[]
            )
            axis.imshow(rows[row][col])

    plt.show()

def show_normalization(data, x, index):
    fig = plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.imshow(mpimg.imread(data[index]))
    plt.title('Original Image')
    plt.subplot(122)
    plt.plot(x[index])
    plt.title('Features')
    fig.tight_layout()
