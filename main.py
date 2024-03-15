import cv2
import numpy as np


def reduce_resolution(image, block_size):

    height, width = image.shape[:2]

    reduced_image = np.zeros_like(image)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):

            block = image[i:i + block_size, j:j + block_size]

            average_pixel_value = np.mean(block)

            reduced_image[i:i + block_size, j:j + block_size] = average_pixel_value

    return reduced_image


def main():

    image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not open or find the image.")
        return

    # Reduce resolution for 3x3 blocks
    reduced_image_3x3 = reduce_resolution(image, 3)

    # Reduce resolution for 5x5 blocks
    reduced_image_5x5 = reduce_resolution(image, 5)

    # Reduce resolution for 7x7 blocks
    reduced_image_7x7 = reduce_resolution(image, 7)

    cv2.imshow('Original Image', image)
    cv2.imshow('Reduced Resolution Image 3x3', reduced_image_3x3)
    cv2.imshow('Reduced Resolution Image 5x5', reduced_image_5x5)
    cv2.imshow('Reduced Resolution Image 7x7', reduced_image_7x7)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
