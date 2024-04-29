import tkinter
import customtkinter
from PIL import Image, ImageTk
import cv2
import numpy as np         
import pyrealsense2 as rs

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        ####### Realsense setup ##############################################################################################
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        self.profile = None#self.pipeline.start(self.config)
        self.colorizer = rs.colorizer()

        # Camera flags
        self.running = False # Global flag to control video capture
        self.roi_start = None
        self.roi_defined = False
        #####################################################################################################################

        ####### Main window setup #############################################################################################
        # configure window
        self.title("LoRDE - Long Range Depth Estimation Framework")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure((1,2), weight=1)
        self.grid_columnconfigure((0), weight=0)
        self.grid_rowconfigure(( 1), weight=1)
        #####################################################################################################################

        ###### Create Sidebar Frame with Widgets ###############################################################################
        #frame
        self.sidebar_frame = customtkinter.CTkFrame(self, width=150, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        #logo label
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="LoRDE 1.0", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        #start video stream button
        self.start_stream = customtkinter.CTkButton(self.sidebar_frame, command=self.start_video, text="Start Video Stream",font=customtkinter.CTkFont(size=16))
        self.start_stream.grid(row=1, column=0, padx=20, pady=10)
        #stop video stream button (disabled by default)
        self.stop_stream = customtkinter.CTkButton(self.sidebar_frame, command=self.stop_video, text="Stop Video Stream",font=customtkinter.CTkFont(size=16), state="disabled")
        self.stop_stream.grid(row=2, column=0, padx=20, pady=10)

        #### radiobuttons for stream type #######
        # radio frame #
        self.radiobutton_frame = customtkinter.CTkFrame(master=self.sidebar_frame, fg_color="transparent")
        self.radiobutton_frame.grid(row=3, column=0, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.radio_var = tkinter.BooleanVar(value=True)
        # radio label #
        self.label_radio_group = customtkinter.CTkLabel(master=self.radiobutton_frame, text="Stream Type:", font=customtkinter.CTkFont(size=18))
        self.label_radio_group.grid(row=0, column=0, columnspan=1, padx=10, pady=10, sticky="")
        # rgb radio button #
        self.radio_button_1 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=True, text="RGB", font=customtkinter.CTkFont(size=16))
        self.radio_button_1.grid(row=1, column=0, pady=10, padx=20, sticky="")
        # depth radio button #
        self.radio_button_2 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=False, text="Depth", font=customtkinter.CTkFont(size=16))
        self.radio_button_2.grid(row=2, column=0, pady=10, padx=20, sticky="s")

        ###### dark/light mode switch button ######
        # appearance mode label #
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w", font=customtkinter.CTkFont(size=16))
        self.appearance_mode_label.grid(row=6, column=0, padx=20, pady=(10, 0))
        # appearance option menu #
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"], command=self.change_appearance_mode_event, font=customtkinter.CTkFont(size=14))
        self.appearance_mode_optionemenu.grid(row=7, column=0, padx=20, pady=(10, 10))
        # set default appearance mode
        self.appearance_mode_optionemenu.set("Dark")
        ######################################################################################################################

        ########### Main frame widgets ########################################################################################
        # welcome message
        self.title_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.title_frame.grid(row=0, column=1, padx=(20, 20), pady=(20, 0), sticky="nsew", rowspan=1, columnspan=1)
        self.title_welcome = customtkinter.CTkLabel(self.title_frame,text="Welcome to LoRDE!\n", font=customtkinter.CTkFont(size=24))
        self.title_welcome.grid(row=0, column=0, padx=(20, 0), pady=(20,0), sticky="nsew")
        #align text left
        self.title_info = customtkinter.CTkLabel(self.title_frame, height=40,text="Where we seek accomplish to improve the long range depth estimation \nof the Intel RealSense by matching similar objects in the near and far plane.\n We use the depth data from the near object to estimate its size. \nFrom this data we can estimate the distance between the two matching objects. ", font=customtkinter.CTkFont(size=14), anchor="w")
        self.title_info.grid(row=1, column=0, padx=(20, 0), pady=(0,20), sticky="nsew")

        # video frame
        self.video_frame = customtkinter.CTkFrame(self)
        self.video_frame.grid(row=1, column=1, padx=(20, 20), pady=(0, 0), sticky="nsew", rowspan=1, columnspan=1)

        self.video_label = customtkinter.CTkLabel(self.video_frame, text="")
        self.video_label.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.video_label.bind("<Button-1>", self.on_mouse_click)  # Mouse click
        self.video_label.bind("<B1-Motion>", self.on_mouse_drag)  # Mouse drag
        self.video_label.bind("<ButtonRelease-1>", self.on_mouse_release)  # Mouse release

        # create slider and progressbar frame
        self.slider_progressbar_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.slider_progressbar_frame.grid(row=2, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.slider_progressbar_frame.grid_columnconfigure(0, weight=1)
        self.slider_progressbar_frame.grid_rowconfigure(4, weight=1)
        
        self.progressbar_1 = customtkinter.CTkProgressBar(self.slider_progressbar_frame)
        self.progressbar_1.grid(row=0, column=0, pady=(10, 10), sticky="nsew")

        #clip frame display
        self.clip_frame = customtkinter.CTkFrame(self, width=250)
        self.clip_frame.grid(row=0, column=2, padx=(0, 20), pady=(20, 0), sticky="nsew", rowspan=2, columnspan=2)

        # clip data frame
        self.clip_frame = customtkinter.CTkFrame(self.clip_frame, fg_color="#3b3b3b")
        self.clip_frame.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")
        #clip label
        self.clip_label = customtkinter.CTkLabel(self.clip_frame, text="Clip Reigion:", font=customtkinter.CTkFont(size=18))
        self.clip_label.grid(row=0, column=0, padx=(20, 0), pady=(20, 10), sticky="nsew")

        self.clip_data = customtkinter.CTkLabel(self.clip_frame, text="", height=40, width=40, fg_color="transparent")
        self.clip_data.grid(row=1, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew", rowspan=2, columnspan=1)
        
        #exit button
        self.main_button_1 = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), text="Exit", command=self.exit)
        self.main_button_1.grid(row=2, column=2, padx=(0, 20), pady=(20, 20), sticky="nsew")

        self.progressbar_1.configure(mode="indeterminnate")
        self.progressbar_1.start()
        #####################################################################################################################
 
    def change_appearance_mode_event(self, new_appearance_mode: str): # Event handler for changing the appearance mode
        customtkinter.set_appearance_mode(new_appearance_mode)

    def start_video(self): # Start the video display
        self.running = True
        self.show_frame()
        self.start_stream.configure(state="disabled")
        self.stop_stream.configure(state="normal")

    
    def stop_video(self):  # Freeze the video display
        self.running = False
        self.start_stream.configure(state="normal")
        self.stop_stream.configure(state="disabled")

    def show_frame(self): # Display the video frame, draw the ROI on top,
        global frame_captured, depth_colormap, depth_image
        if self.running:

            frames = self.pipeline.wait_for_frames() # Align frames
            align = rs.align(rs.stream.color)
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame() # Get frame data
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
           
            frame_captured = color_image
            
            if not self.radio_var.get(): # toggle between depth and color image
                frame_captured_copy = depth_colormap
            else:
                frame_captured_copy = frame_captured.copy()
            

            # # get aspect ratio of the image
            color_colormap_dim = color_image.shape
            aspect_ratio = color_colormap_dim[1] / color_colormap_dim[0]
            
            display_width = 480
            display_height = int(display_width / aspect_ratio)  

            # Calculate scaling factor
            scale_x = display_width / color_colormap_dim[1]
            scale_y = display_height / color_colormap_dim[0]
            
            if self.roi_defined and self.roi_coords:
                x1, y1, x2, y2 = self.roi_coords
                cv2.rectangle(frame_captured_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw ROI in green

            cv_rgb = cv2.cvtColor(frame_captured_copy, cv2.COLOR_BGR2RGB)
            
            img = Image.fromarray(cv_rgb)
            my_image = customtkinter.CTkImage(light_image=img,dark_image=img, size=(display_width, display_height))
            self.video_label.configure(image=my_image)
            self.video_label.image = my_image
            self.video_label.after(10, self.show_frame)

    def display_roi(self): # Display the selected region of interest, and save the images to jpgs
        global frame_captured
        if self.roi_coords and frame_captured is not None:
            x1, y1, x2, y2 = self.roi_coords
            roi = frame_captured[y1-10:y2+10, x1-10:x2+10]
            depth_roi = depth_colormap[y1-10:y2+10, x1-10:x2+10]
            #depth_roi_data = depth_image[y1-10:y2+10, x1-10:x2+10] #<-- ANDREW: this is the depth data for the ROI (HxWx1) to use for depth estimation
            print("x1: {}, y1: {}, x2: {}, y2: {}".format(x1, y1, x2, y2))
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            roi_dim = roi.shape
            aspect_ratio = roi_dim[1] / roi_dim[0]
            roi_image = Image.fromarray(roi) #<-- ANDREW: this is where we save the rgb roi image (HxWx3) to use for sift
            depth_roi = Image.fromarray(depth_roi) #<-- ANDREW: this is the depth colormap roi image (HxWx3), no useful for calculations but for good for visualization
            roi_image.save('roi_image.jpg')
            depth_roi.save('depth_roi.jpg')
            roi_photo = customtkinter.CTkImage(light_image=roi_image,dark_image=roi_image, size=(250, int(250 / aspect_ratio)))
            self.clip_data.configure(image=roi_photo)  # Keep reference
            self.clip_data.image = roi_photo
        else: 
            print("No ROI coordinates defined, please select a region of interest first")

    def on_mouse_click(self, event): # Sets the starting coordinates of the ROI
        self.roi_start = (event.x, event.y)
        print("Mouse clicked at:", self.roi_start)

    def on_mouse_drag(self,event): # Updates the changing ROI coordinates
        if self.roi_start is not None:
            x1, y1 = self.roi_start
            x2, y2 = event.x, event.y
            self.roi_coords = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            self.roi_defined = True

    def on_mouse_release(self,event):  # Update final coordinates of the ROI
        self.on_mouse_drag(event)  # Update coordinates
        self.display_roi()  # Display the selected ROI
        self.roi_defined = False  # Reset flag after processing
        self.roi_start = None  # Reset start coordinates

    def exit(self): # Qxit the application
        self.destroy()
if __name__ == "__main__":
    app = App()
    app.mainloop()
