import customtkinter as ctk
import subprocess
import threading
import os

# Set appearance mode and color theme
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Color Detection Hub")
        self.geometry("500x450")
        self.resizable(False, False)

        # Title Label
        self.title_label = ctk.CTkLabel(
            self, text="Color Detection App", font=ctk.CTkFont(size=28, weight="bold")
        )
        self.title_label.pack(pady=(30, 10))

        self.subtitle_label = ctk.CTkLabel(
            self, text="Select a tool to launch:", font=ctk.CTkFont(size=16)
        )
        self.subtitle_label.pack(pady=(0, 30))

        # Buttons Frame
        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # App 1 Button
        self.btn_app1 = ctk.CTkButton(
            self.button_frame, 
            text="1. Image Color Picker", 
            font=ctk.CTkFont(size=18),
            height=50,
            command=lambda: self.run_script("1.py")
        )
        self.btn_app1.pack(pady=10, fill="x")
        
        self.desc1 = ctk.CTkLabel(self.button_frame, text="Double-click an image to pick a color.", text_color="gray")
        self.desc1.pack(pady=(0, 15))

        # App 2 Button
        self.btn_app2 = ctk.CTkButton(
            self.button_frame, 
            text="2. Basic Color Masks (Webcam)", 
            font=ctk.CTkFont(size=18),
            height=50,
            command=lambda: self.run_script("2.py")
        )
        self.btn_app2.pack(pady=10, fill="x")
        
        self.desc2 = ctk.CTkLabel(self.button_frame, text="Real-time blue, red, and green mask filters.", text_color="gray")
        self.desc2.pack(pady=(0, 15))

        # App 3 Button
        self.btn_app3 = ctk.CTkButton(
            self.button_frame, 
            text="3. Advanced Object Tracking", 
            font=ctk.CTkFont(size=18),
            height=50,
            command=lambda: self.run_script("3.py")
        )
        self.btn_app3.pack(pady=10, fill="x")
        
        self.desc3 = ctk.CTkLabel(self.button_frame, text="Real-time multi-color tracking with bounding boxes.", text_color="gray")
        self.desc3.pack(pady=(0, 15))

    def run_script(self, script_name):
        # Run in a separate thread so it doesn't freeze the GUI
        def target():
            subprocess.run(["python", script_name])
        
        thread = threading.Thread(target=target)
        thread.start()

if __name__ == "__main__":
    app = App()
    app.mainloop()
