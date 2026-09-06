import cv2
import numpy as np
import os
import random

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def blur_image(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def change_brightness_contrast(image, brightness, contrast):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

def stretch_zoom_image(image, fx, fy):
    return cv2.resize(image, None, fx=fx, fy=fy)

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(51, 101):
        filename = f"ds_{i}.jpg"
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(input_path):
            image = cv2.imread(input_path)
            
            transformation = random.choice(['rotate', 'blur', 'brightness_contrast', 'stretch_zoom'])
            
            if transformation == 'rotate':
                angle = random.uniform(-30, 30)
                transformed_image = rotate_image(image, angle)
            elif transformation == 'blur':
                kernel_size = random.choice([3, 5, 7])
                transformed_image = blur_image(image, kernel_size)
            elif transformation == 'brightness_contrast':
                brightness = random.randint(-50, 50)
                contrast = random.uniform(0.5, 1.5)
                transformed_image = change_brightness_contrast(image, brightness, contrast)
            elif transformation == 'stretch_zoom':
                fx = random.uniform(0.8, 1.2)
                fy = random.uniform(0.8, 1.2)
                transformed_image = stretch_zoom_image(image, fx, fy)
            
            cv2.imwrite(output_path, transformed_image)
            print(f"Processed {filename} with {transformation} and saved to {output_path}")

if __name__ == "__main__":
    input_directory = "data/raw"
    output_directory = "data/processed"
    process_images(input_directory, output_directory)