# Imports:
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read an image:
def readImage(imagePath):
    # Loads image:
    inputImage = cv2.imread(imagePath)
    # Checks if image was successfully loaded:
    if inputImage is None:
        print("readImage>> Error: Could not load Input image.")
    return inputImage

# Defines a re-sizable image window:
def showImage(imageName, inputImage, delay=0):
    cv2.namedWindow(imageName, flags=cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(delay)

# Writes an PGN image:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)

# Set image path:
path = "/Users/ernestoguevara/Desktop/VisionRobotos/img"
fileName = "/noisyDominoes.png"

# Read image:
inputImage = readImage(path + fileName)

# Show Image:
showImage("Input Image", inputImage)

# Set the kernel size:
kernelSize = [(3,3), (5,5), (7,7)]
for i in range (len(kernelSize)):
    #Average Filtered:
    averageFiltered = cv2.blur(inputImage, kernelSize[i])
    showImage("Average Filter " + str(kernelSize[i][0]) + "x" + str(kernelSize[i][1]), averageFiltered)
    average_path = path + "average" + str(kernelSize[i][0]) + "x" + str(kernelSize[i][1]) + ".png"
    cv2.imwrite(average_path, averageFiltered)

    #Gaussian Filtered:
    gaussianFiltered = cv2.GaussianBlur(inputImage, kernelSize[i], 0)
    showImage("Gaussian Filter " + str(kernelSize[i][0]) + "x" + str(kernelSize[i][1]), gaussianFiltered)
    gaussian_path = path + "gaussian" + str(kernelSize[i][0]) + "x" + str(kernelSize[i][1]) + ".png"
    cv2.imwrite(gaussian_path, gaussianFiltered)

# Parte 2: Filtrado Adaptativo utilizando un filtro mediana #
kSize = [3, 5, 7]
for i in range (len(kSize)):
    #Median Filtered:
    medianFiltered = cv2.medianBlur(inputImage, kSize[i])
    showImage("Median Filter " + str(kSize[i]) + "x" + str(kSize[i]), medianFiltered)


# Parte 3: Máscara de Outliers

#Grayscale Conversion:
grayscaleImage = cv2.cvtColor(inputImage , cv2.COLOR_BGR2GRAY)

# Create histogram:
histogram, binLimits = np.histogram(grayscaleImage, bins=256)

# Configure and draw the histogram:
plt.figure()
plt.title("Grayscale histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")

# Set the X Axis values:
plt.xlim([0.0, 255.0])
# Plot histogram:
plt.plot(binLimits[0:-1], histogram)
# Show histogram:
plt.show()
thresholdValue=1
_, lowMask = cv2.threshold(grayscaleImage, thresholdValue, 255, cv2.THRESH_BINARY_INV)
showImage("Low Outliers Mask", lowMask)

thresholdValue=254
_, highMask = cv2.threshold(grayscaleImage, thresholdValue, 255, cv2.THRESH_BINARY)
showImage("High Outliers Mask", highMask)

outliersMask = lowMask + highMask
showImage("outliersMask", outliersMask)
outliers_path = path + "outliersMask.png"
cv2.imwrite(outliers_path, outliersMask)

# Parte 4: Implementación del filtro adaptativo


#Valor maximo y minimo
originalHeight, originalWidth = inputImage.shape[:2]
for y in range (originalHeight):
    for x in range (originalWidth):
        if outliersMask[y,x] == 255:
            inputImage[y,x]=medianFiltered[y, x]
showImage("Adaptative Filter", inputImage)


#Parte 5: Detección de círculos

# The function receives threshold as numpy arrays (not lists)
hsvImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)

lowThreshold = np.array([24, 100, 100]) # H, S, V
highThreshold = np.array([32, 255, 255]) # H, S, V

hsvMask = cv2.inRange(hsvImage, lowThreshold, highThreshold)
showImage ("Binary Mask [Yellow circles]", hsvMask)

circles = cv2.HoughCircles(hsvMask, cv2.HOUGH_GRADIENT, dp=5, minDist=20, param1=100, param2=70, minRadius=0, maxRadius=17)
circles = np.round(circles[0, :]).astype("int")

circlesCounter = 1

for (x, y, r) in circles:
    # Draw the circle:
    # The function receives:
    # Image to draw the circles on, center of circle, radius, BGR tuple color, thickness
    cv2.circle(inputImage, (x, y), r, (0, 255, 0), 2)
    cv2.circle(inputImage, (x, y), 2, (0, 0, 255), 5)

    # Draw text:
    # The function receives:
    # Image to draw text on, string to draw, text position, font, font scale, font color, text thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (0, 0, 0)  # black
    fontScale = 0.5
    fontThickness = 2
    cv2.putText(inputImage, str(circlesCounter), (x, y), font, fontScale, fontColor, fontThickness)

    # Increase the circles counter:
    circlesCounter += 1

# Show image
showImage("Circles Detected", inputImage)
circlesDetected_path = path + "circlesDetected.png"
cv2.imwrite(circlesDetected_path, inputImage)