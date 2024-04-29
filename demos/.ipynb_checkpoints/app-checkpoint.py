import tkinter as tk
from tkinter import Label, Button
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np         
import pyrealsense2 as rs

import utils
from utils import lorde_from_roi

## This displays each frame from the webcam in a window
def show_frame():
    global lmain, running, frame_captured, depth_colormap, depth, roi_defined, roi_coords, pipeline, config
    if running:

        #align frames
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        # Get frame data
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            print("Failed to capture frame")
            running = False
            return
        
        # Convert images to numpy arrays
        depth = np.asanyarray(depth_frame.get_data())
        depth_image = np.asanyarray(depth)
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_image.shape
        color_colormap_dim = color_image.shape

        frame_captured = color_image
        
        if not show_color:
            frame_captured_copy = depth_colormap
        else:
            frame_captured_copy = frame_captured.copy()
        
        
        if roi_defined and roi_coords:
            x1, y1, x2, y2 = roi_coords
            
            cv2.rectangle(frame_captured_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw ROI in green

        
        cv_rgb = cv2.cvtColor(frame_captured_copy, cv2.COLOR_BGR2RGB)
        
        # get aspect ratio of the image
        aspect_ratio = color_colormap_dim[1] / color_colormap_dim[0]

        # resize the image to fit the window size
        #cv_rgb = cv2.resize(cv_rgb, (int(800), int(800 / aspect_ratio)))
        
        img = Image.fromarray(cv_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)


        

def display_roi():
    global roi_coords, frame_captured, depth
    if roi_coords and frame_captured is not None:
        x1, y1, x2, y2 = roi_coords
        roi = frame_captured[y1:y2, x1:x2]
        depth_roi = depth_colormap[y1:y2, x1:x2]
        print("x1: {}, y1: {}, x2: {}, y2: {}".format(x1, y1, x2, y2))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_image = Image.fromarray(roi)
        depth_roi = Image.fromarray(depth_roi)
        roi_image.save('roi_image.jpg')
        depth_roi.save('depth_roi.jpg')
        roi_photo = ImageTk.PhotoImage(image=roi_image)
        roi_label.imgtk = roi_photo  # Keep reference
        roi_label.configure(image=roi_photo)

        out = lorde_from_roi(roi_coords, frame_captured, depth)
        matched_boxes = out['matched_boxes']
        computed_depths = out['computed_depths']

        full_frame = cv2.cvtColor(frame_captured, cv2.COLOR_BGR2RGB)
        for i, box in enumerate(matched_boxes):
            cv2.rectangle(full_frame, box[0], box[1], (0, 225, 0), 2)
        
            computed_depth = computed_depths[i][0]
            realsense_depth = computed_depths[i][1]
        
            if computed_depth:
                computed_depth = "%.1f" % computed_depth
                cv2.putText(full_frame, computed_depth, (box[0][0], box[0][1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            
            if realsense_depth:
                realsense_depth = "%.1f" % realsense_depth
                cv2.putText(full_frame, realsense_depth, (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        #full_frame.save('full_frame.jpg')
        full_frame = Image.fromarray(full_frame)
        full_frame = ImageTk.PhotoImage(image=full_frame)
        full_frame_label.imgtk = full_frame
        full_frame_label.configure(image=full_frame)
    else: 
        print("No ROI coordinates defined, please select a region of interest first")

def start_video(): # Start the video display
    global running
    running = True
    show_frame()

def stop_video():  # Properly stop the video display
    global cap, running
    running = False

def switch_stream():
    global show_color
    show_color = not show_color

def on_mouse_click(event): # gets the starting coordinates of the ROI
    global roi_start
    roi_start = (event.x, event.y)
    print("Mouse clicked at:", roi_start)


def on_mouse_drag(event): # keeps track of the changing ROI coordinates
    global roi_coords, roi_defined, roi_start
    if roi_start is not None:
        x1, y1 = roi_start
        x2, y2 = event.x, event.y
        roi_coords = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        roi_defined = True
        # print("ROI Coordinates:", roi_coords)

def on_mouse_release(event):
    global roi_coords, roi_defined, roi_start
    on_mouse_drag(event)  # Update final coordinates
    display_roi()  # Display the selected ROI
    roi_defined = False  # Reset flag after processing
    roi_start = None  # Reset start coordinates

#realsense setup
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

profile = pipeline.start(config)
colorizer = rs.colorizer()


# Main window setup
window = tk.Tk()
window.title("Long Range Depth Estimation Framework")
window.minsize(1280, 800)


# Camera and flags
running = False # Global flag to control video capture
roi_start = None
roi_defined = False
show_color = True

title = Label(window, text="Welcome to LoRDE!", font=("Lexend", 32))
subtitle = Label(window, text="Long Range Depth Estimation. \nWhere we accomplish long range depth estimation \nby using similar objects in the near and far plane", font=("Lexend", 16))
video_button = Button(window, text="Start Video Stream", height=2, width=20, command=start_video)
#swtich button
switch_button = Button(window, text="Switch Stream", height=2, width=20, command=switch_stream)
stop_button = Button(window, text="Stop Video Stream", height=2, width=20, command=stop_video)

lmain = Label(window)
roi_label = Label(window)  # Label to display the ROI
full_frame_label = Label(window)

title.grid(row=0, column=0)
subtitle.grid(row=1, column=0)
video_button.grid(row=2, column=0)
switch_button.grid(row=3, column=0)
stop_button.grid(row=4, column=0)
lmain.grid(row=5, column=0)
roi_label.grid(row=1, column=1, rowspan=3)
full_frame_label.grid(row=5, column=1)

#lmain.pack(side='left')
#roi_label.pack(side='bottom')  # Pack ROI label below the main video label
#full_frame_label.pack(side='right')

#title.pack()
#subtitle.pack()
#video_button.pack()
#switch_button.pack()
#stop_button.pack()

#lmain.pack()

# Bind mouse events
lmain.bind("<Button-1>", on_mouse_click)  # Mouse click
lmain.bind("<B1-Motion>", on_mouse_drag)  # Mouse drag
lmain.bind("<ButtonRelease-1>", on_mouse_release)  # Mouse release

# Start the GUI
window.mainloop()