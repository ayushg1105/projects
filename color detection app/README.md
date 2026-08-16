# Advanced Color Detection

A complete computer vision project with a modern graphical interface that allows users to accurately detect and track colors in real-time. This project leverages OpenCV, Pandas, and CustomTkinter to provide multiple color-tracking tools.

## Features

### 1. Main Screen GUI (`main.py`)
A modern, dark-themed dashboard built with CustomTkinter that serves as a central hub. It allows you to effortlessly launch any of the color detection modules below in separate background threads.

### 2. Picture Color Picker (`1.py`)
Code that detects colors from a static picture (`colorpic.jpg`). Simply double-click anywhere on the image, and the program will cross-reference the RGB values with our 800+ color dataset (`colors.csv`) to tell you the exact name of the color you clicked.

### 3. Basic Mask Filters (`2.py`)
Code that detects red, green, and blue colors in different windows. It applies real-time HSV color masking to your webcam feed, isolating specific colors while blacking out the rest of the environment.

### 4. Advanced Bounding Box Tracker (`3.py`)
Code that detects colors and draws bounding boxes around them in one central window.
Features 5 interactive tracking modes via a Trackbar:
- **Mode 0 (All Colors):** Detects any brightly colored object and accurately matches it against the dataset in real-time using vectorized calculations.
- **Modes 1, 2, 3 (Specific Colors):** Highly robust tracking for Red, Green, and Blue objects individually.
- **Mode 4 (Off):** View the raw webcam feed.

## Requirements

To run this project, make sure you have Python installed, and install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The required packages are:
- `opencv-python`
- `numpy`
- `pandas`
- `customtkinter`

## Running the Project

After installing the requirements, simply launch the main GUI:
```bash
python main.py
```
From there, you can click on any tool to begin tracking!

## Bibliography & References
- [Flat UI Color Picker](http://www.flatuicolorpicker.com)
- [Bestech Blogs](https://www.bestech.com.au/blogs)
- [Python Official Documentation](https://docs.python.org)
- [DigitalOcean Tutorials](https://www.digitalocean.com)
- [W3Schools](https://www.w3schools.com)
- [MathWorks](https://www.mathworks.com)
