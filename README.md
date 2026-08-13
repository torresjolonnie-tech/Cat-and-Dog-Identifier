# Cat and Dog Identifier

## About

The Cat and Dog Identifier is a real-time computer vision project that uses a camera to detect and identify cats and dogs. The project runs on an NVIDIA Jetson device and uses **NanoOWL** for object detection and **OpenCV** to display the camera feed and detection results.

The program checks the live camera feed and determines whether a cat or dog is visible. When a cat or dog is detected, the name of the detected animal is displayed on the screen.

## Features

* Detects cats and dogs in real time
* Uses a live camera feed
* Uses NanoOWL for AI-based object detection
* Uses OpenCV for camera input and display
* Runs on NVIDIA Jetson hardware
* Displays the detected animal on the camera window
* Shows the normal camera feed when no cat or dog is detected
* Press `q` to close the program

## How It Works

1. The camera starts and captures live video.
2. Each video frame is sent to the object detection model.
3. NanoOWL analyzes the frame for objects.
4. The program checks whether a cat or dog has been detected.
5. If a cat or dog is detected, its label is displayed on the screen.
6. If neither is detected, the normal camera feed is shown.
7. The program continues until the user presses `q`.

## Technology Used

### Python

Python is used to create the main detection program and control the camera and AI model.

### OpenCV

OpenCV is used to:

* Capture frames from the camera
* Display the video
* Add text to the video
* Detect when the user presses `q`

The output window is named:

`NanoOWL Cat/Dog Detector`

### NanoOWL

NanoOWL is used as the AI object detection system. It analyzes the camera frames and identifies objects such as cats and dogs.

### NVIDIA Jetson

The project is designed to run on an NVIDIA Jetson device, allowing the AI model to use the device's GPU capabilities for real-time detection.

### Jetson Containers

The project can be run using NVIDIA's Jetson Containers environment. The setup shown in the project uses the `dustynv/nanoowl` container to provide the required NanoOWL environment.

## Camera Detection

The program uses the camera to continuously capture images. When a cat or dog is detected, the program creates a display frame and places the detected label in the center of the screen.

For example:

`Cat`

or

`Dog`

If no supported animal is detected, the program simply displays the original camera feed.

## Running the Project

The project is intended to run on an NVIDIA Jetson device with the required NanoOWL environment.

A Jetson Container can be started with:

```bash
sudo jetson-containers run --workdir /opt/nanoowl ${autotag nanoowl}
```

After starting the environment, the Python detection program can be run from the project directory.

Example:

```bash
python3 animal_green_screen.py
```

The exact command may be different depending on the location and name of the Python file.

## Controls

| Key | Action           |
| --- | ---------------- |
| `q` | Exit the program |

## Example Output

When the camera sees a cat:

`Cat`

When the camera sees a dog:

`Dog`

When there is no cat or dog:

The normal camera feed is displayed.

## Project Structure

The main detection program is contained in:

```text
animal_green_screen.py
```

Other Python files in the project can be used for testing, recognition, or additional functionality.

## Requirements

* NVIDIA Jetson device
* Camera
* Python 3
* OpenCV
* NanoOWL
* NVIDIA Jetson Containers
* Compatible Jetson software and drivers

## Purpose

The purpose of this project is to demonstrate real-time AI object detection on an NVIDIA Jetson device. It combines a camera, computer vision, and an AI detection model to recognize cats and dogs from a live video feed.

## Future Improvements

Possible improvements include:

* Add detection boxes around the animal
* Display the detection confidence
* Detect more types of animals
* Add a green-screen effect when an animal is detected
* Improve detection speed
* Save detected images or video
* Add support for multiple animals at the same time
* Create a simple web interface for the detector

## License

This project is intended for learning, experimentation, and computer vision development.
