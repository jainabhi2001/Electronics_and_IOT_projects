import numpy as np
import cv2
import serial

########## THRESHOLD ###########
right_th = 30
left_th = 30
area_th = 100

######## Communicating to ARDUINO ########
######## Initialise SERIAL ########
# ser = serial.Serial('/dev/ttyACM0', 9600)

# initalize the cam
cap = cv2.VideoCapture(0)

lower = np.array([7, 170, 147], dtype="uint8")        #Thresholds for the line -yellow
upper = np.array([28, 232, 231], dtype="uint8")

#black lower-[72,24,33]
#black upper - [146,83,145]

#black lower-[54,52,48]
#black upper - [156,255,255]

#lower = np.array([0, 0, 0], dtype="uint8")        #Thresholds for the line -black
#upper = np.array([180, 255, 30], dtype="uint8")
    
while True:
    # image=frame
    ret, image = cap.read()
    image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    original = image.copy()
    # print("Show track !!")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image, lower, upper)                      #MASK for the 2 side path lanes - yellow 
    
    # get dimensions of image
    dimensions = image.shape

    # # height, width, number of channels in image
    # height = img.shape[0]
    width = image.shape[1]
    # channels = img.shape[2]

    cnts,hierarchy = cv2.findContours(mask, 1, cv2.CHAIN_APPROX_NONE)
    if len(cnts) <=0:
        print ("I don't see the line")
        print ("Turn Right")
        a='r'
        # ser.write(a.encode())

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > area_th:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0


            cv2.line(original,(cx,0),(cx,720),(25,56,4),1)
            cv2.line(original,(0,cy),(1280,cy),(25,56,4),1)
            cv2.circle(original, (cx, cy), 5, (48, 39, 210), 2)
            
            cv2.drawContours(original, cnts, -1, (0,255,0), 1)
            if cx <= ((width)//2 - left_th) and cx>0:
                print ("Turn Left!")
                a='l'
                # ser.write(a.encode())
            
            elif (cx < ((width//2)+ right_th) and cx > ((width//2) - left_th)):
                print ("On Track!")
                a='f'
                # ser.write(a.encode())
            
            elif cx >= ((width//2 + right_th )):
                print ("Turn Right")
                a='r'
                # ser.write(a.encode())
            
            elif cx==0:
                print ("I don't see the line")
                print ("Turn Right")
                a='r'
                # ser.write(a.encode())

            cv2.line(original,((width//2)- left_th,0),((width//2)- left_th,dimensions[0]),(255,0,125),2)    # Parallel to x-axis
            cv2.line(original,((width//2)+ right_th,0),((width//2)+ right_th,dimensions[0]),(255,0,125),2)    # Parallel to y-axis

            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
    if (cv2.waitKey(10) & 0xFF == ord('q') ):
        # ser.close()
        cap.release()
        cv2.destroyAllWindows()
        break
    cv2.imshow('original', original)
