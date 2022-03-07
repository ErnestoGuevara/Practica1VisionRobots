from matplotlib import pyplot as plt
import numpy as np
import cv2
import os

#Current pixels
currentPixelsArray = []
#Read image
def readImage(imagePath):
    #Loads image via OpenCV
    inputImpage= cv2.imread(imagePath)
    #Check if image was succesfully loaded
    if inputImpage is None:
        print("Error: (-215: Assertion Failed)")
    return inputImpage

#Defines a re-sizable image window
def showImage(windowName, inputImage):
    #Create the window
    cv2.namedWindow(windowName, flags=cv2.WINDOW_GUI_NORMAL)
    #Show Image
    cv2.imshow(windowName, inputImage)
    #Wait keyboard event: 
    cv2.waitKey(0)
#Image path
rootDir = "/Users/ernestoguevara/Desktop/VisionRobotos/img"

filename = "koopa.png"

filepath = os.path.join(rootDir,filename)

inputImage = cv2.imread(filepath)

#Show image 
showImage("Input Image", inputImage)
print("Image dtype",inputImage.dtype)
#Color en posición 130,129
print("Valor de Color posición 130,129: " + str(inputImage[130,129]) )
print("B: " +  str(inputImage[130,129][0]))
print("G: " +  str(inputImage[130,129][1]))
print("R: " +  str(inputImage[130,129][2]))


#Grayscale Conversion:
grayScaleImage = cv2.cvtColor(inputImage , cv2.COLOR_BGR2GRAY)
showImage("Gray Image", grayScaleImage)

originalHeight, originalWidth = inputImage.shape[:2]

#Manual inverse
manualInverse = np.zeros((originalHeight, originalWidth), np.uint8)



for y in range(originalHeight):
    for x in range(originalWidth):
        currentPixel = grayScaleImage[y,x]
        #print("Y: " + str(y) + " " + "X: " + str(x))
       
        #Inverse GrayPixel
        inverseGrayPixel = 255 - currentPixel
        manualInverse[y, x] = inverseGrayPixel

        currentPixelsArray.append(currentPixel)
        #Valor en x=130 y=129
        if x== 130 and y==129:
            print("Valor de pixel en X=130, Y=129: "+ str(currentPixel))
        #if x == 0 and y==0:
            #print("Punto inical en " + "Y: " + str(y) + " " + "X: " + str(x))
#Valor maximo y minimo
maximo = max(currentPixelsArray)
minimo = min(currentPixelsArray)
print("Valor máximo: "+ str(maximo) + " Valor minimo: "+ str(minimo))

showImage("Inverse image (per pixel)", manualInverse)



