from helper import readFromcsv, retreive_image, groundTruth
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import skimage as ski
from skimage import data
from skimage.color import rgb2hsv
import cv2

imgs = retreive_image([x["image_id"] for x in readFromcsv()])


def hsv_cov():
    for img in imgs[:1]:

         # Read image as BGR format
        #image = cv2.imread(img)
        image = img

        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for red color in HSV (adjust based on needs)
        lower_red = np.array([0, 255, 255], np.uint8)
        upper_red = np.array([179, 255, 255], np.uint8)

        # Create a mask to identify red pixels
        mask = cv2.inRange(hsv_image, lower_red, upper_red)

        # Apply bitwise AND with original image to extract red pixels
        red_image = cv2.bitwise_and(image, image, mask=mask)
        plt.imshow(red_image)
        plt.show()

        return red_image

        # Example usage
    

        # Display the red-extracted image using OpenCV (modify for other libraries)
    

        # rgb_img = img
        # HSV = rgb2hsv(rgb_img);
        # # Define masks for red pixels in HSV color space
        # red_hue_low = 0;        
        # red_hue_high = 0.1;   

        # #Create mask for pixels with red hue
        # red_mask = (HSV[:,:,1] >= red_hue_low) & (HSV[:,:,1] <= red_hue_high);

        # # Apply the red mask
        # filtered_RGB = uint8(zeros(size(rgb_img)));  
        # filtered_RGB[:,:,1] = rgb[:,:,1] * uint8(red_mask); 
        # filtered_RGB(:,:,2) = 0; 
        # filtered_RGB(:,:,3) = 0; 

        # # Display the filtered image
        # imshow(filtered_RGB);

        # # #lower maskkk
        # lower_red = np.array([0, 50, 50])
        # upper_red = np.array([10, 255, 255])
        # mask0 = cv2.inRange(hsv_img, lower_red, upper_red)

        #         # upper mask (170-180)
        # lower_red = np.array([170,50,50])
        # upper_red = np.array([180,255,255])
        # mask1 = cv2.inRange(hsv_img, lower_red, upper_red)

        # # join my masks
        # mask = mask0+mask1

        # # set my output img to zero everywhere except my mask
        # output_img = img.copy()
        # output_img[np.where(mask==0)] = 0

        # # or your HSV image, which I *believe* is what you want
        # output_hsv = hsv_img.copy()
        # output_hsv[np.where(mask==0)] = 0

        plt.imshow(output_hsv)
        plt.title("Color Image")
        plt.show()

        # img_copy = np.copy(img)
        # img_copy[(img_copy[:,:,0] > 50) | (img_copy[:,:,1] > 50) | (img_copy[:, :, 2] < 90) ]=0
        # img_2 = np.hstack(( cv2.resize(img, (650, 500)), cv2.resize(img_copy, (650, 500)) ))
        # plt.imshow(img_2)
        # plt.title("Color Image VS Color Extracted Image")
        # plt.show()
        # pix_val = (img.data())
        # print(pix_val)
        # rgb_img = img
        # hsv_img = rgb2hsv(rgb_img)
        
        # hue_img = hsv_img[:, :, 0]
        # print( hsv_img)

        # # Define thresholds for red hue values
        # lower_red = 0.95
        # upper_red = 0.05

        # for pixel in hsv_img:
        #     if()


        # red_mask = ((hue_img >= 0.95) and (hue_img <= 1) )or (hue_img >= 0) and (hue_img <= 0.05)

        # red_pixels = np.zeros_like(hsv_img)
        # red_pixels[red_mask] = hsv_img[red_mask]

        # red_masked_img = np.zeros_like(rgb_img)
        # red_masked_img[red_mask] = rgb_img[red_mask]    
        # # Plot the original and red masked images
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # axes[0].imshow(rgb_img)
        # axes[0].set_title('Original RGB Image')
        # axes[0].axis('off')

        # axes[1].imshow(red_masked_img)
        # axes[1].set_title('Red Masked Image')
        # axes[1].axis('off')

        # plt.show()
        # Threshold the hue channel to isolate red pixels
#         binary_img = ((hue_img >= 0.95) and (hue_img <= 1) )or (hue_img >= 0) and (hue_img <= 0.05)

#         plt.figure(figsize=(4, 3))
#         plt.imshow(binary_img, cmap='gray')
#         plt.title(f"Red pixels in image {i}")
#         plt.axis('off')

#         plt.tight_layout()
#         plt.show()

hsv_cov()

# from helper import readFromcsv, retreive_image, groundTruth
# import numpy as np
# import matplotlib.pyplot as plt
# import skimage as ski
# from skimage.color import rgb2hsv

# # Step 1: Retrieve the image
# imgs = retreive_image([x["image_id"] for x in readFromcsv()])

# # Step 2: Convert the image to HSV color space
# imgs_hsv = rgb2hsv(imgs)

# # Step 3: Define the range for red color in HSV
# # Red hue values wrap around, so you need to handle both ends
# lower_red = np.array([0, 0.5, 0.5])  # Lower bound for red
# upper_red = np.array([0.1, 1, 1])     # Upper bound for red

# # Step 4: Mask the HSV image to extract red pixels
# mask_red = cv2.inRange(imgs_hsv, lower_red, upper_red)

# # Step 5: Store the red pixels in a NumPy array
# red_pixels = imgs[mask_red]

# # Optionally, convert back to RGB if needed
# red_pixels_rgb = ski.color.hsv2rgb(red_pixels)

# # Display the original image
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(imgs)
# plt.title('Original Image')

# # Display the image with only red pixels
# plt.subplot(1, 2, 2)
# plt.imshow(red_pixels_rgb)
# plt.title('Extracted Red Pixels')

# plt.show()
