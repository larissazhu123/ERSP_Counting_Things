import cv2
import matplotlib.pyplot as plt

# Read the original image
img = cv2.imread(r'.\data\images_384_VarV2\5027.jpg')

# Convert BGR image to RGB for displaying with Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display original image
plt.imshow(img_rgb)
plt.title('Original')
plt.axis('off')
plt.show()

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection

# Display Sobel Edge Detection Images
plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X')
plt.axis('off')
plt.show()

plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y')
plt.axis('off')
plt.show()

plt.imshow(sobelxy, cmap='gray')
plt.title('Sobel X Y using Sobel() function')
plt.axis('off')
plt.show()

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection

# Display Canny Edge Detection Image
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')
plt.show()
