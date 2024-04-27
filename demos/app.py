import tkinter as tk
from tkinter import Label, Button
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np         
import pyrealsense2 as rs

## This displays each frame from the webcam in a window
def show_frame():
    global lmain, running, frame_captured, depth_colormap, roi_defined, roi_coords, pipeline, config
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
        depth_image = np.asanyarray(depth_frame.get_data())
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
    global roi_coords, frame_captured
    if roi_coords and frame_captured is not None:
        x1, y1, x2, y2 = roi_coords
        roi = frame_captured[y1-10:y2+10, x1-10:x2+10]
        depth_roi = depth_colormap[y1-10:y2+10, x1-10:x2+10]
        print("x1: {}, y1: {}, x2: {}, y2: {}".format(x1, y1, x2, y2))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_image = Image.fromarray(roi)
        depth_roi = Image.fromarray(depth_roi)
        roi_image.save('roi_image.jpg')
        depth_roi.save('depth_roi.jpg')
        roi_photo = ImageTk.PhotoImage(image=roi_image)
        roi_label.imgtk = roi_photo  # Keep reference
        roi_label.configure(image=roi_photo)
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
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

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
lmain.pack(side='left')
roi_label.pack(side='right')  # Pack ROI label below the main video label
title.pack()
subtitle.pack()
video_button.pack()
switch_button.pack()
stop_button.pack()

lmain = Label(window)
lmain.pack()

# Bind mouse events
lmain.bind("<Button-1>", on_mouse_click)  # Mouse click
lmain.bind("<B1-Motion>", on_mouse_drag)  # Mouse drag
lmain.bind("<ButtonRelease-1>", on_mouse_release)  # Mouse release

# Start the GUI
window.mainloop()
