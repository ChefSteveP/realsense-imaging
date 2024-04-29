import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures

from PIL import Image

def color_selection_mask(image, target_hsv=list):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) 
    target_hsv = np.tile(target_hsv, (480, 640, 1))
    
    distance = np.zeros(img_hsv.shape)
    h_distance = np.minimum(np.abs(img_hsv[:,:,0:1] - target_hsv[:,:,0:1]), 180 - np.abs(img_hsv[:,:,0:1] - target_hsv[:,:,0:1]))
    sv_distance = img_hsv[:,:,1:] - target_hsv[:,:,1:]
    
    distance[:,:,0:1] = h_distance
    distance[:,:,1:] = sv_distance
    
    distance = np.linalg.norm(distance, axis=2)
    distance = 255 * distance / distance.max()
    distance = 255 - distance.astype(np.uint8)

    return distance


def auto_canny(image, sigma=0.33):
    # Compute the median of the single channel pixel intensities
    v = np.median(image)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged


def multiscale_template_matching(image, template_canny, max_scale=4, num_scales=61, do_visualize=False):    
    found = None
    template = template_canny #auto_canny(template) # Just pass in canny of template to save time
    (tH, tW) = template.shape[:2]
    
    # Loop over the scales of the image
    for scale in np.linspace(1, max_scale, num_scales)[::-1]:
        # Resize the image according to the scale, and keep track of the ratio of the resizing
        resized = cv2.resize(image, (0,0), fx=scale, fy=scale)
        r = image.shape[1] / float(resized.shape[1])
        # If the resized image is smaller than the template, then break from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
    
        # Detect edges in the resized, grayscale image and apply template matching to find the template in the image
        edged = auto_canny(resized)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        
        # Check to see if the iteration should be visualized
        if do_visualize:
            # Draw a bounding box around the detected region
            clone = np.dstack([edged, edged, edged])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (
                maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            plt.imshow(clone)
            plt.show()
    
        # If we have found a new maximum correlation value, then update the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    
    # Unpack the bookkeeping variable and compute the (x, y) coordinates of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    return {'box': ((startX, startY), (endX, endY)), 'scale': r}


def find_matching_boxes(image, template):
    # Parameters and their default values
    MAX_MATCHING_OBJECTS = 2

    # Find keypoints and descriptors for the template

    matched_boxes = []
    scales = []
    matching_img = image.copy()
    template_canny = auto_canny(template)

    for i in range(MAX_MATCHING_OBJECTS):
        # Find bounding boxes using template matching
        output = multiscale_template_matching(matching_img, template_canny)
        box = output['box']
        scale = output['scale']
        
        matched_boxes.append(box)
        scales.append(scale)
        
        if i == MAX_MATCHING_OBJECTS - 1:
            break
        
        # Create a mask and fill the matched area with near neighbors
        corners = np.array([[[box[0][0], box[0][1]]],
                            [[box[0][0], box[1][1]]],
                            [[box[1][0], box[1][1]]],
                            [[box[1][0], box[0][1]]]])
        mask = np.ones_like(matching_img) * 255
        cv2.fillPoly(mask, [np.int32(corners)], 0)
        mask = cv2.bitwise_not(mask)
        matching_img = cv2.inpaint(matching_img, mask, 3, cv2.INPAINT_TELEA)

        #plt.imshow(mask, 'gray')
        #plt.show()
        #plt.imshow(matching_img, 'gray')
        #plt.show()

    return matched_boxes, scales


def depth_from_boxes(depth, boxes):
    # Assume first box is always closest box, for now only works with 2 boxes
    fov_per_pix = 0.1491
    depth_scale = 0.0010000000474974513
    box1, box2 = boxes

    # Get pixel dimentions and depth of closest box
    height_close_pixels = abs(box1[1][1] - box1[0][1])
    width_close_pixels = abs(box1[1][0] - box1[0][0])
    depth_close = np.median(depth[box1[0][1]:box1[1][1], box1[0][0]:box1[1][0]])  * depth_scale

    print('Height of close object in pixels', height_close_pixels)
    print('Width of close object in pixels', width_close_pixels)
    print('Depth of close object:', depth_close)

    # Compute angular dimentions of closest box
    angular_size_height_close = height_close_pixels * fov_per_pix
    angular_size_width_close = width_close_pixels * fov_per_pix

    # Convert angular dimentions and stereo depth to physical dimentions of closest box
    height_close = 2 * depth_close * np.tan(np.deg2rad(angular_size_height_close)/2)
    width_close = 2 * depth_close * np.tan(np.deg2rad(angular_size_width_close)/2)

    print('')
    print('Height of close object', height_close)
    print('Width of close object', width_close)

    # Get pixel dimentions of farther box
    height_far_pixels = abs(box2[1][1] - box2[0][1])
    width_far_pixels = abs(box2[1][0] - box2[0][0])
    depth_far = np.median(depth[box2[0][1]:box2[1][1], box2[0][0]:box2[1][0]]) * depth_scale

    print('')
    print('Height of far object in pixels', height_far_pixels)
    print('Width of far object in pixels', width_far_pixels)

    # Compute angular dimentions of farther box
    angular_size_height_far = height_far_pixels * fov_per_pix
    angular_size_width_far = width_far_pixels * fov_per_pix

    # Compute depth angular dimentions known physical dimentions
    depth_from_height_far = height_close / (2 * np.tan(np.deg2rad(angular_size_height_far)/2))
    depth_from_width_far =  width_close / (2 * np.tan(np.deg2rad(angular_size_width_far)/2))
    depth_mean_far = np.mean([depth_from_height_far, depth_from_width_far])

    print('')
    print('Depth from height:', depth_from_height_far)
    print('Depth from width:', depth_from_width_far)
    print('Average depth:', np.mean([depth_from_height_far, depth_from_width_far]))
    
    return [(None, depth_close), (depth_mean_far, depth_far)]


def lorde_from_roi(roi_coords, bgr_frame, depth):
    x1, y1, x2, y2 = roi_coords
    img = color_selection_mask(bgr_frame, [0, 255, 255])
    ref_img = img[y1:y2, x1:x2]
    matched_boxes, scales = find_matching_boxes(img, ref_img)
    computed_depths = depth_from_boxes(depth, matched_boxes)
    return {'matched_boxes': matched_boxes, 'computed_depths': computed_depths}



