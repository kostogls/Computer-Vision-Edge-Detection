import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read video object using opencv
cap = cv2.VideoCapture("C:/Users/User/garden_sif.y4m")
i = 0
# Cycle through pictures
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow("frame", frame)
    else:
        print("Video playback is complete!")
        break

    # Stop playing
    cv2.imwrite('kang' + str(i) + '.jpg', frame)
    i += 1
    key = cv2.waitKey(25)
    if key == 27:  # Button esc
        break

# Free resources
cap.release()
cv2.destroyAllWindows()


# perform convolution for Sobel filter
def convolution(image, kernel, average=False, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    print("Kernel Shape : {}".format(kernel.shape))

    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    print("Output Image size : {}".format(output.shape))

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()

    return output


# Perform sobel filter for edge detection
def sobel_edge_detection(image, filter, verbose=False):
    new_image_x = convolution(image, filter, verbose)

    if verbose:
        plt.imshow(new_image_x, cmap='gray')
        plt.title("Horizontal Edge")
        plt.show()

    new_image_y = convolution(image, np.flip(filter.T, axis=0), verbose)

    if verbose:
        plt.imshow(new_image_y, cmap='gray')
        plt.title("Vertical Edge")
        plt.show()

    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()

    return gradient_magnitude

# filter for convolution with every image
filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


img_array = []
files = glob.glob('*.jpg')
files = sorted(files, key=lambda x: int(x[4:-4]))
print(files)

for filename in files:
    img = cv2.imread(filename)
    h, w, col = img.shape
    size = (h, w)
    new = sobel_edge_detection(img, filter, verbose=True)
    h2, w2 = new.shape
    new_size = (h2, w2)
    img_array.append(new)

print(new_size)

out = cv2.VideoWriter('output_1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (w2, h2), False)

for i in range(len(img_array)):
    out.write(np.uint8(img_array[i]))
out.release()


img_array2 = []
files = glob.glob('*.jpg')
files = sorted(files, key=lambda x: int(x[4:-4]))
print(files)
for filename in files:
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    size2 = (height, width)
    img_array2.append(gray)

new_img_array = []
# Images iteration
for i in range(len(img_array2)):
    new_img = np.zeros((height, width))
    # Frame iteration
    # for j in range(i-1, i+2):
    #     if i < 0 or i > 114:
    #         continue
    #     # Kernel iteration
    for k in range(height):
        for g in range(width):
            if k < 1 or g < 1 or k > height-1 or g > width - 1:
                new_img[k][g] = 0
                continue
            for n in range(3):
                for m in range(3):
                    new_img[k][g] += img_array2[i][k][g]
            new_img[k][g] /= 27
    new_img_array.append(new_img)


# image = np.array(new_img_array)

plt.imshow(new_img_array[1], cmap=plt.cm.gray)
plt.show()


out2 = cv2.VideoWriter('output_2.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height), False)

for i in range(len(new_img_array)):
    out2.write(np.uint8(new_img_array[i]))
out2.release()
