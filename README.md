# Anaconda
If you want to use just Python and not Colab, I highly recommend installing Anaconda. Anaconda is a necessity for data science and you should learn how to use it because it will change your life.

Anaconda Download
(https://www.anaconda.com/products/individual)

Anaconda Cheat Sheet
https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)

# Install Anaconda Environment
conda env create -f environment.yml

conda activate cvpj1

OR

Pip Setup

# Install Modules Pip
pip3 install -r requirements.txt

Run program

Runs program, will write output images to output directory

python3 project.py

# How To Run: Colab
Open project.py in a text editor and then copy/paste into a new Colab Notebook.

DO NOT ADD ANY NEW CODE CELLS OR TEXT CELLS!!!
Once coding is complete go to the File tab in Colab, then Download, choose ".py".

# Transformation

code in Python 3 to create many transformed versions of an image. 

This way of creating transformed version of an image is a process, often called image augmentation, is used to increase the amount of training data for
current deep learning-based approaches to object recognition.

Type of transformations to be implemented:

Let the input image be W x H sized.

There are two ways to output the transformed image:

(A) choose the portion of the output plane where the output image is mapped. This is done by the
inverse_warp function. Note the output image could be different from W x H in
this case.

(B) choose W x H portion of the output plane with the top-left corner remaining at (0, 0), i.e. we
choose the portion of the output plane that coincides with the input image (the inverse_warp2
function).

  1. Translation X % shift, Y % shift, rest will be black. X and Y shifts are user specified, % is
     based on the total number of rows and cols in the image. Positive percentages are shifts to
     the right (or down), and negative ones are to the left (or up). Use method (A) to specify the
     output image.
  2. Rotate theta degrees. Theta is the angle of 2D rotation specified in degrees. Positive values
     denote counterclockwise rotation and negative values denote clockwise rotation. Use
     method (B) to specify the output image.
  3. Scale % the image. Greater than 100% denotes expansion, and less than 100% is
     contraction. Use method (B) to specify the output image.
  4. Affine, A, is a user specified vector of 6 parameters a_00, a_11, a_01, a_10, t_x, t_y. Use
     method (B) to specify the output image.
  5. Projective, H, is a user specified vector of 8 parameters h_00, h_11, h_12, h_10, h_11,
     h_12, h_21, h_22, (h_23=1) . Use method (B) to specify the output image.
  6. Contrast and brightness modulation of the L-channel of the input image input using "a" and
     "b". The output image will be the same size as the input and in RGB format.
  7. Gamma correction of the L-channel of the input image input using "a" and "b". The output
     image will be the same size as the input and in RGB format.
  8. Histogram equalization of the L-channel of the input image input using "a" and "b". The
     output image will be the same size as the input and in RGB format.
  9. Compute the mean image and the standard deviation image of a collection of images.
  10. Batch normalize an image given a collection of images.


# Estimation of Transformation images
Python 3 code to estimate the transformation between two images

Steps:

1. Detect image point features in each of the images

2. Match features detected in the images to create a list of possible matches (M) of
the image features

3. Write a function to compute the least square estimate of the transformation
parameters (P) given a small set of matching points (m)

  The type of transformations could be (i) affine, (ii) 2D rotation and
translation, and (iii) 2D perspective (or homography)

4. Estimate transformation parameters given the matches using RANSAC (Random
Sample Consensus). The algorithm works by returning the best estimate among
many random samples of matches (m) - the minimal subset of correspondences
needed - and selecting the best parameter estimate based on the error on the rest.

# Multiple Camera geometry

The estimation of 3D geometry using multiple-camera setups. Develop a deep understanding of epipolar geometry and 
constraints among images collected with multiple calibrated cameras

For this part, we used WILDTRACK dataset

Part I:

Write a function to read each of the cameras' intrinsic and extrinsic calibration information from 
the xml files and create: 
  
    (i) a plot of the camera locations, with respect to the world coordinates, 
        on the horizontal plane. For each camera, show the view direction (i.e., camera z-axis).
    (ii) in the images from each camera, overlay the plot of the world-coordinate axes if the axes are visible
         in the view. Note the world coordinate origin might not be seen in all the views.

         
Part II:
 
Write a function that will draw epipolar lines in a collection of images taken 
with calibrated cameras. The function will take a pixel location for one of the camera images and 
draw the epipolar lines on the images from the 6 other cameras.

Part III:

Write a function that will take a pair of corresponding points between any two camera images and 
draw the epipolar lines for these two points in all the other 5 camera images. Note that the 
intersection of the two epipolar lines should be near the corresponding point in those images.

Part IV:

Write a function that will take corresponding points between any set of (> 2) camera images and 
draw the epipolar lines for these points in all the other camera images. Note that the intersection 
of the epipolar lines should be near the corresponding feature in images.

Part V:

Write code to estimate the height of the person. Not expect the process to be fully 
automated. Here I manually mark the head's pixel location and the person's feet in each camera 
image. 

These should then be fed to a 3D estimation function to estimate the height of the person. 

You can use any off-the-shelf image editing software (preview in mac or Irfanview for windows) to 
locate pixel coordinates. Consider ± 1pixel variation in the location of the top and bottom points 
on the person and estimate the variation in height.

 
