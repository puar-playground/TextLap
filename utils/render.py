import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
import gdown
import matplotlib.pyplot as plt

def is_image_too_white(image, threshold=200):
    """
    Examines the color of non-transparent pixels in an image to determine if it is too white.
    
    Parameters:
    - image (PIL.Image.Image): The input PIL image.
    - threshold (int): The threshold value for determining if the image is too white (default is 200).
    
    Returns:
    - bool: True if the average color of non-transparent pixels is too white, otherwise False.
    """
    image = image.convert("RGBA")
    pixels = image.getdata()

    non_transparent_pixels = [pixel for pixel in pixels if pixel[3] != 0]

    if not non_transparent_pixels:
        return False  # No non-transparent pixels in the image

    avg_color = (
        sum(pixel[0] for pixel in non_transparent_pixels) / len(non_transparent_pixels),
        sum(pixel[1] for pixel in non_transparent_pixels) / len(non_transparent_pixels),
        sum(pixel[2] for pixel in non_transparent_pixels) / len(non_transparent_pixels)
    )
    
    avg_brightness = sum(avg_color) / 3  # Average brightness of RGB components
    
    return avg_brightness >= threshold

def plot_histogram(data, bins=10, save_dir=None, title="Histogram", xlabel="Value", ylabel="Frequency"):
    """
    Plots a histogram from a list of numbers.

    Parameters:
    - data (list): List of numbers to create the histogram from.
    - bins (int): Number of bins in the histogram (default is 10).
    - title (str): Title of the histogram (default is "Histogram").
    - xlabel (str): Label for the x-axis (default is "Value").
    - ylabel (str): Label for the y-axis (default is "Frequency").
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if save_dir is not None:
        plt.savefig(save_dir)
        plt.close()
    # else:    
        # plt.show()
    

def count_transparent_pixels(image):
    """
    Computes the number of transparent pixels and the total number of pixels in a given PIL image.

    Parameters:
    - image (PIL.Image.Image): The input PIL image.

    Returns:
    - tuple: A tuple containing the number of transparent pixels and the total number of pixels in the image.
    """
    image = image.convert("RGBA")
    pixels = image.getdata()

    transparent_pixel_count = sum(1 for pixel in pixels if pixel[3] == 0)
    total_pixel_count = len(pixels)
    
    return transparent_pixel_count, total_pixel_count


def convert_transparent_to_white(image):
    # Ensure the image has an alpha channel
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Create a new image with a white background
    white_bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
    
    # Paste the input image onto the white background
    white_bg.paste(image, (0, 0), image)
    
    # Convert the result to RGB (drop alpha channel)
    result_image = white_bg.convert('RGB')
    
    return result_image


def draw_images_on_canvas(bounding_boxes, images, canvas_size, resize=True):
    """
    Draw images in the corresponding bounding box on a canvas.

    Parameters:
    bounding_boxes (list of list of int): List of bounding boxes with [x1, x2, y1, y2] coordinates.
    images (list of PIL.Image): List of PIL images to be drawn.
    canvas_size (tuple of int): Size of the canvas (width, height).

    Returns:
    PIL.Image: The resulting canvas with images drawn.
    """
    # Create a blank canvas
    canvas = Image.new('RGB', canvas_size, 'white')
    
    # Iterate over bounding boxes and images
    for bbox, img in zip(bounding_boxes, images):
        x1, y1, x2, y2 = bbox
        # x1, x2 = int(x1* canvas_size[0]), int(x2* canvas_size[0])
        # y1, y2 = int(y1* canvas_size[1]), int(y2* canvas_size[1])

        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        # Resize image to fit the bounding box
        img_resized = img.resize((int(x2 - x1), int(y2 - y1)))

        
        
        # Paste the image onto the canvas at the specified position
        canvas.paste(img_resized, (x1, y1), img_resized)

    canvas = canvas.convert('RGB')
    # canvas.show()

    if resize:
        return canvas.resize((256, 256))
    else:
        return canvas


def combine_images_with_captions_matplotlib(image1, image2, output_path):
    """
    Combine two PIL images side by side using matplotlib, with captions "True Layout" and "Generated Layout" below them.

    Args:
        image1 (PIL.Image): The first image (True Layout).
        image2 (PIL.Image): The second image (Generated Layout).
        output_path (str): The file path where the combined image will be saved.
    """
    
    # Convert PIL images to arrays for matplotlib
    image1_array = np.array(image1)
    image2_array = np.array(image2)
    
    # Create a figure and set of subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))

    # Show the first image (True Layout) on the left
    axes[0].imshow(image1_array)
    axes[0].set_title('True Layout', fontsize=12)
    axes[0].axis('off')  # Hide axis

    # Show the second image (Generated Layout) on the right
    axes[1].imshow(image2_array)
    axes[1].set_title('Generated Layout', fontsize=12)
    axes[1].axis('off')  # Hide axis

    # Adjust layout to fit everything
    plt.tight_layout()

    # Save the figure to the specified output path
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)

    # Show the result (optional)
    plt.show()
