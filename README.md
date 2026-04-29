# 3DCV-practice

This repository contains the code and resources for the 3D Computer Vision (3DCV) course at Hochschule Furtwangen. The accompanying website can be found at [https://uhahne.github.io/3DCV/](https://uhahne.github.io/3DCV/).

# Excercise 1

The purpose of this set of scripts is to capture images of a chess board pattern and extract intrinsic calibration parameters from them. Then, another script uses those calibration data to project a virtual coordinate system onto that same checkerboard pattern in real time.

## Capturing images

To capture images, run

```
python3 scripts/ex1/capture.py
```

In this interactive script, you can:
- Delete all previous images by pressing 'D'
- Capture the current image by pressing 'S'
- Quit by pressing 'Q'

The images are stored into `captured_images`. You should take images with a checkerboard pattern at different locations in the image.

## Computing calibration parameters

The images captured in the last step are all used to compute the calibration parameters. You can execute this by doing

```
python3 scripts/ex1/detect_intrinsics.py
```

The output is stored in `calibration.npz`.
You can test if this worked correctly by executing

```
python3 scripts/ex1/undistort.py
```

This should display the captured image, alongside the corrected, processed image.

## Projecting the coordinate system

If the previous steps were followed and the camera calibration is setup, you can run

```
python3 scripts/ex1/project_coordinate_system.py
```

This will overlay a 3D coordinate system over the captured video stream in real time.

# Excercise 2

The purpose of this excercise is to measure the height of a cup, relative to a known reference object (bottle).
To do this, execute

```
python3 scripts/ex2/height_measurement.py
```

You'll get a loupe window, which is just a zoomed in version of the area around your mouse for precise picking. You'll also get the main window, in which you must select the following points in order:

1. Start and end point of the first of the two parallel ground reference lines (i.e one table edge)
2. Start and end point of the second parallel ground reference lines (i.e the other, parallel table edge)
3. Base and Top of the bottle
4. Base and Top of the cup

All relevant information such as helper lines will be displayed in the main image after the relevant points are selected. When all points have been selected, the estimated height will be output next to the projected lines. It is also output on the terminal.

Using this script, I get an estimated cup height of ~11.3 cm.
