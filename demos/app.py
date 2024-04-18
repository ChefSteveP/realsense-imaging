import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2

def show_frame():
    global cap, lmain, running, frame_captured
    if running:
        ret, frame_captured = cap.read()
        if ret:
            #frame = cv2.flip(frame_captured, 1)
            if roi_defined and roi_coords:
                x1, y1, x2, y2 = roi_coords
                cv2.rectangle(frame_captured, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw ROI in green
            cv_rgb = cv2.cvtColor(frame_captured, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            lmain.after(10, show_frame)
        else:
            print("Failed to capture frame")
            running = False

def display_roi():
    global roi_coords, frame_captured
    if roi_coords and frame_captured is not None:
        x1, y1, x2, y2 = roi_coords
        roi = frame_captured[y1:y2, x1:x2]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_image = Image.fromarray(roi)
        roi_image.save('roi_image.jpg')
        roi_photo = ImageTk.PhotoImage(image=roi_image)
        roi_label.imgtk = roi_photo  # Keep reference
        roi_label.configure(image=roi_photo)

def start_video():
    global cap, running
    cap = cv2.VideoCapture(0)  # 0 is usually the built-in webcam
    if not cap.isOpened():
        print("Failed to open video stream")
        return
    running = True
    show_frame()

def stop_video():
    global cap, running
    running = False
    if cap:
        cap.release()


def on_mouse_click(event):
    global roi_start
    roi_start = (event.x, event.y)

def on_mouse_drag(event):
    global roi_coords, roi_defined
    x1, y1 = roi_start
    x2, y2 = event.x, event.y
    roi_coords = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    roi_defined = True

def on_mouse_release(event):
    global roi_coords, roi_defined
    # if the the x2 is less than x1, swap them
    # if roi_coords[0] > roi_coords[2]:
    #     roi_coords = (roi_coords[2], roi_coords[1], roi_coords[0], roi_coords[3])
    # # if the y2 is less than y1, swap them
    # if roi_coords[1] > roi_coords[3]:
    #     roi_coords = (roi_coords[0], roi_coords[3], roi_coords[2], roi_coords[1])

    on_mouse_drag(event)  # Update final coordinates
    display_roi()  # Display the selected ROI
    roi_defined = False  # Reset flag after processing


# Main window setup
window = tk.Tk()
window.title("Long Range Depth Estimation Framework")
window.minsize(800, 600)

# Camera and flags
cap = cv2.VideoCapture(0)
running = False # Global flag to control video capture
frame_captured = None
roi_start = None
roi_coords = None
roi_defined = False

# GUI Elements
title = Label(window, text="Welcome to LoRDE!", font=("Lexend", 32))
subtitle = Label(window, text="Long Range Depth Estimation. \nWhere we accomplish long range depth estimation \nby using similar objects in the near and far plane", font=("Lexend", 16))
video_button = Button(window, text="Start Video Stream", height=2, width=20, command=start_video)
stop_button = Button(window, text="Stop Video Stream", height=2, width=20, command=stop_video)
lmain = Label(window)
roi_label = Label(window)  # Label to display the ROI
lmain.pack(side='left')
roi_label.pack(side='right')  # Pack ROI label below the main video label
title.pack()
subtitle.pack()
video_button.pack()
stop_button.pack()

lmain = Label(window)
lmain.pack()

# Bind mouse events
lmain.bind("<Button-1>", on_mouse_click)  # Mouse click
lmain.bind("<B1-Motion>", on_mouse_drag)  # Mouse drag
lmain.bind("<ButtonRelease-1>", on_mouse_release)  # Mouse release

# Start the GUI
window.mainloop()
