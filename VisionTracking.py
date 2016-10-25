import cv2
import numpy as np
import os
import time
import math
from collections import Counter
lowThresh = np.array([40,190,200])
highThresh = np.array([90,255,255])

def main():
    aim = False
    capWebcam = cv2.VideoCapture(0);
    capWebcam.set(cv2.CAP_PROP_FRAME_WIDTH, 320.0)
    capWebcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240.0)

    if capWebcam.isOpened() == False:
        os.system("pause")
        return
    frameCount = 1
    distArrayRound = []
    #Get each frame and analyze 1 per iteration
    while cv2.waitKey(1) != 27 and capWebcam.isOpened():
        blnFrameReadSuccessfully, imgOriginal = capWebcam.read()
        if not blnFrameReadSuccessfully or imgOriginal is None:
            print "error; frame not read"
            os.system("pause")
            break
        start = time.time()
        #Convert to HSV and filter based on HSV values
        imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
        imgThresh = cv2.inRange(imgHSV, lowThresh, highThresh)
        #Minimize small errors in order to help with corner detection
        imgThresh = minimizeError(imgThresh)
        #Find all determined contours with 8 corners (desired object)
        properShapes = sortContoursByCorners(imgThresh, 8)
        #Find the largest contour with 8 corners (remove random false detections)
        goal = sortContoursByArea(properShapes)
        #Distance calculations based on experimental data of frame size to real size
        if goal != None:
            #Calculate center of goal and draw on circle for ease of use
            xMin, xMax, yMin, yMax = boundriesOfContour(goal)
            centerX = (xMax-xMin)/2 + xMin
            centerY = (yMax-yMin)/2 + yMin
            cv2.circle(imgThresh, (centerX, centerY), 20, (90, 255, 180), -1)
            #Calculate width and height in pixels and estimate real distance based on experimental data
            width = xMax-xMin
            height = yMax-yMin
            dist = 9793.69398*(math.pow(width, -1.063924091))
            #Compare y coordinate of the bottom of the goal to the y coordinate of the center of the frame
            #Output value determining if shot is too high, too low, or on target
            yCenter = 120.0
            writeVal1 = ""
            if yMin > (yCenter + 6):
                writeVal1 = "high"
            if yMin < (yCenter - 6):
                writeVal1 = "low"
            if yMin < (yCenter + 6) and yMin > (yCenter - 6):
                writeVal1 = "true"
            writeVal2 = ""
            #Compare x coordinates through same process as with y coordinates
            if centerX < (160 - 6):
                writeVal2 = "right"
            if centerX > (160 + 6):
                writeVal2 = "left"
            if centerX > 154 and centerX < 166:
                writeVal2 = "true"
            #Write values to txt file that is read from main processor
            try:
                target = open("/home/pi/code/listen/secondStats.txt", "w")
                target.write(writeVal1 + "," + writeVal2)
            except IOError:
                print("Stats file not found")
                break
            #Every 5 frames, take the most common distance and write to file, in order to minimze false detections
            if frameCount != 5:
                distArrayRound.append(math.ceil(dist))
                frameCount = frameCount + 1
            else:
                distCounter = Counter(distArrayRound)
                distCounted = distCounter.most_common(1)
                dist = distCounted[0][0]
                count = distCounted[0][1]
                #Calculate x coordinate distance from the center of the frame to the center of the goal
                #Approximate a real distance using proportions and calculate an angle to turn
                if count >= 2 and dist <= 264:
                    xOff = 160-centerX
                    xOffReal = (20.0/width)*(xOff)
                    if (abs(xOffReal/dist) <= 1):
                        angle = math.asin(xOffReal/dist)*180.0/math.pi
                        if xOffReal < 1.0:
                            aim = True
                        #Write values to txt file that is read from main processor
                        try:
                            target = open("/home/pi/code/listen/stats.txt", "w")
                            target.write(str(dist) + "," + str(angle) + "," + str(aim))
                        except IOError:
                            print("Stats file not found")
                            break
                #Restart the 5 frame count of distances
                del distArrayRound[:]
                frameCount = 1
        else:
            #If no object detected, write false, false to file
            try:
                target = open("/home/pi/code/listen/secondStats.txt", "w")
                target.write("false,false")
            except IOError:
                print("Stats file not found")
                break
        #Draw the identified goal onto frame to see easily
        cv2.drawContours(imgThresh, [goal], 0,(90,255,180),-1)
        end = time.time()
    return

def minimizeError(frame):
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    kernel = np.ones((5,5),np.uint8)
    frame = cv2.erode(frame, kernel, iterations = 1)
    frame = cv2.dilate(frame, kernel, iterations = 1)
    return frame

def sortContoursByCorners(frame, numCorners):
    _, contours, _ = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    matchingContours = []
    approxs = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02*cv2.arcLength(contour, True), True)
        approxs.append(approx)
        if len(approx) is numCorners:
            matchingContours.append(contour)
    return matchingContours

def sortContoursByArea(contours):
    largestContour = None
    largestArea = 0
    for contour in contours:
        if cv2.contourArea(contour) > largestArea:
            largestArea = cv2.contourArea(contour)
            largestContour = contour
    return largestContour

def boundriesOfContour(contour):
    xMin = tuple(contour[contour[:,:,0].argmin()][0])
    xMax = tuple(contour[contour[:,:,0].argmax()][0])
    yMin = tuple(contour[contour[:,:,1].argmin()][0])
    yMax = tuple(contour[contour[:,:,1].argmax()][0])
    return xMin[0], xMax[0], yMin[1], yMax[1]

if __name__ == "__main__":
    main()
