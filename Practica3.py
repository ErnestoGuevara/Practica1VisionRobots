import numpy as np
import cv2


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


# Writes an PNG image:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)
    



# Set image path:
path = "/Users/ernestoguevara/Desktop/VisionRobotos/img/"
#Parte 1: Lectura de imagenes
fileNameList=["peaceHand.png","closedHand.png"]

#Lista de imagenes (Part 5)
imagesList = []

for i in range(len(fileNameList)):
    fileName = fileNameList[i]
    # Read image:
    inputImage = readImage(path+fileName)
    # Show Image:
    showImage("Input Image", inputImage)
    

    #Parte 2: Segmentando las manos
    blue,green,red = cv2.split(inputImage)
    showImage("Blue Image", blue)
    showImage("Green Image", green)
    showImage("Red Image", red)

    bluePath = path + "BlueImage.png"
    cv2.imwrite(bluePath,blue)
    greenPath = path + "GreenImage.png"
    cv2.imwrite(greenPath,green)
    redPath = path + "RedImage.png"
    cv2.imwrite(redPath,red)

    #OTSU
    _, binaryImage = cv2.threshold(red, 0, 255, cv2.THRESH_OTSU)
    showImage("Binary Image", binaryImage)
    binaryPath = path + "binaryImage.png"
    cv2.imwrite(binaryPath,binaryImage)

    #Parte 3: Afinando la mascara binaria
    morphoKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #print(morphoKernel)
    iterationsList=[1,2,3,4]
    for i in range(len(iterationsList)):
        filteredImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, morphoKernel, iterations=iterationsList[i])
        showImage("Filtered Image "+ str(iterationsList[i]),filteredImage)
        filteredPath = path + "closed-"+str(iterationsList[i])+".png"
        cv2.imwrite(filteredPath,filteredImage)
            
    handMask = filteredImage
    
    #Parte 4: Calculando Bounding Rectangle
    contours, hierarchy = cv2.findContours(handMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


    for c in contours:
        # Compute blob area:
        blobArea = cv2.contourArea(c)
        # Print the blob area:
        print("Blob Area: "+str(blobArea))
        #Obtaining boundingRect
        x,y,w,h = cv2.boundingRect(c)
            
        print("x,y,w,h:",x,y,w,h)

    blobRectangle= [x,y,w,h]
    print(blobRectangle)
    boundingRectangle = cv2.rectangle(inputImage,(x,y), (w+x,h+y), (20, 255, 57), 2)
    showImage("Bounding rectangle", boundingRectangle)

    #Parte 5: Identificación de gestos

    imagesList.append( (blobArea, inputImage) )
    
    # Sort list from largest to smallest first element of each tuple:
    imagesList = sorted(imagesList, key=lambda tup: tup[0], reverse=True) 
    handNames = ["Peace", "Closed"]
    for j in range(len(imagesList)):
        #Nombre de la imagen:
        imageName = handNames[j]
        #Area de blob
        print(handNames[j] + " Blob Area: " + str(imagesList[j][0]))
        #ShowImage:
        showImage(imageName,imagesList[j][1])
        if (imageName == "Peace"):
            peacePath = path + "handDetected_Peace.png"
            cv2.imwrite(peacePath, imagesList[j][1])
        elif (imageName == "Closed"):
            closePath = path + "handDetected_Closed.png"
            cv2.imwrite(closePath, imagesList[j][1])

 

        




    









