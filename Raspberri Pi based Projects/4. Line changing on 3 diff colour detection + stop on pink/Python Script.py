import numpy as np
import cv2
import serial

########## THRESHOLD ###########
right_th = 20
left_th = 20
area_th = 100

"""
green_exp = 'l'         # CLOCKWISE
violet_exp = 'r'        # CLOCKWISE
pink_exp = 'rrrr'         # CLOCKWISE
"""

green_exp = 'rf'        # ANTI_CLOCKWISE
violet_exp = 'll'       # ANTI_CLOCKWISE
pink_exp = 'lllllllll'        # ANTI_CLOCKWISE


boolOppo=False
f_count = 0

g_count = 0

v_count = 0

stop=False

######## Communicating to ARDUINO ########
######## Initialise SERIAL ########
ser = serial.Serial('/dev/ttyACM0', 9600)

# initalize the cam
cap = cv2.VideoCapture(0)

#yellow_lower = np.array([7, 170, 147], dtype="uint8")       # Thresholds for the line -yellow
#yellow_upper = np.array([28, 232, 231], dtype="uint8")
yellow_lower = np.array([6, 50, 87], dtype="uint8")       # Thresholds for the line -yellow
yellow_upper = np.array([27, 255, 231], dtype="uint8")

# pink_lower = np.array([155,148,168], np.uint8)              # Thresholds for pink spot to move opposite
# pink_upper = np.array([175,168,248], np.uint8)
pink_lower = np.array([155,90,168], np.uint8)              # Thresholds for pink spot to move opposite
pink_upper = np.array([175,168,248], np.uint8)

green_lower = np.array([72, 100, 52], dtype="uint8")            # Thresholds for green spot to shift right
green_upper = np.array([92, 203, 187], dtype="uint8")
#green_lower = np.array([72, 183, 51], dtype="uint8")            # Thresholds for green spot to shift right
#green_upper = np.array([92, 203, 131], dtype="uint8")
#green_lower = np.array([68, 142, 74], dtype="uint8")            # Thresholds for green spot to shift right
#green_upper = np.array([88, 162, 154], dtype="uint8")


violet_lower = np.array([121,116,53], dtype="uint8")        #Thresholds for the line -yellow
violet_upper = np.array([141,255,133], dtype="uint8")
# violet_lower = np.array([116,245,51], dtype="uint8")        #Thresholds for the line -yellow
# violet_upper = np.array([136,265,131], dtype="uint8")

while True:
    ret, image = cap.read()
    image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    original = image.copy()
		
    ###### YELLOW ###############
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(image, yellow_lower, yellow_upper)           #MASK for the path lanes - yellow 
    
    # get dimensions of image
    dimensions = image.shape
    width = image.shape[1]

    cnts,hierarchy = cv2.findContours(yellow_mask, 1, cv2.CHAIN_APPROX_NONE)
    if len(cnts) <=0:
        if(boolOppo==False):
            print ("I don't see the line")
            a='rl'
            ser.write(a.encode())
        else:
            if (f_count > 30) :
                stop=True
            else:
                print ("Move straight towards center")
                f_count += 1
                a='f'
                ser.write(a.encode())
     

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
                ser.write(a.encode())

            elif (cx < ((width//2)+ right_th) and cx > ((width//2) - left_th)):
                print ("On Track!")
                a='f'
                ser.write(a.encode())

            elif cx >= ((width//2 + right_th )):
                print ("Turn Right")
                a='r'
                ser.write(a.encode())
            
            elif cx==0:
                print ("m00 is 0")
                print ("Turn Right")
                a='r'
                ser.write(a.encode())

            cv2.line(original,((width//2)- left_th,0),((width//2)- left_th,dimensions[0]),(255,0,125),2)      # Parallel to x-axis
            cv2.line(original,((width//2)+ right_th,0),((width//2)+ right_th,dimensions[0]),(255,0,125),2)    # Parallel to y-axis

            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)

    kernal = np.ones((5, 5), "uint8")
    
    pink_mask = cv2.inRange(image, pink_lower, pink_upper)         # MASK for the pink spot to stop
    pink_mask = cv2.dilate(pink_mask, kernal)
    cnt1, hierarchy1 = cv2.findContours(pink_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic1, contour1 in enumerate(cnt1):
        ######## Get AREA of PINK Contour ########
        area = cv2.contourArea(contour1)
        if(area > 100):
            x, y, w, h = cv2.boundingRect(contour1)
            original = cv2.rectangle(original, (x, y),(x + w, y + h),(203, 192, 255), 2)
            cv2.putText(original, "Pink Stop", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(203,192,255))
            print("\nPink detected\n")
            boolOppo=True
            a=pink_exp                                                  # MOVE OPPOSITE
            ser.write(a.encode())

    green_mask = cv2.inRange(image, green_lower, green_upper)       # MASK for the green spot to stop
    green_mask = cv2.dilate(green_mask, kernal)
    cnts2, hierarchy2 = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic2, contour2 in enumerate(cnts2):
        ######## Get AREA of green Contour ########
        area = cv2.contourArea(contour2)
        if(area > 100):
            x, y, w, h = cv2.boundingRect(contour2)
            original = cv2.rectangle(original, (x, y),(x + w, y + h),(0, 0, 0), 2)
            cv2.putText(original, "green Stop", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 0))
            print("\nGreen detected\n")
            if (g_count > 10) :
                a='l'
                ser.write(a.encode())
            else:
                #print ("Move straight towards center")
                g_count += 1
                a=green_exp
                ser.write(a.encode())
            # a=green_exp                                                  # TURN RIGHT
            # ser.write(a.encode())

    violet_mask = cv2.inRange(image, violet_lower, violet_upper)    # MASK for the Violet spot to stop
    violet_mask = cv2.dilate(violet_mask, kernal)
    cnts3, hierarchy3 = cv2.findContours(violet_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic3, contour3 in enumerate(cnts3):
        ######## Get AREA of Violet Contour ########
        area = cv2.contourArea(contour3)
        if(area > 100):
            x, y, w, h = cv2.boundingRect(contour3)
            original = cv2.rectangle(original, (x, y),(x + w, y + h),(25, 0, 127), 2)
            cv2.putText(original, "Violet Stop", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(25, 0, 127))
            print("\nViolet detected\n")
            if (v_count > 10) :
                a='rf'
                ser.write(a.encode())
            else:
                #print ("Move straight towards center")
                v_count += 1
                a=violet_exp
                ser.write(a.encode())
            # a=violet_exp                                                  # TURN LEFT
            # ser.write(a.encode())

    if (cv2.waitKey(10) & 0xFF == ord('q') ) or stop:
        ser.close()
        cap.release()
        cv2.destroyAllWindows()
        break
    cv2.imshow('original', original)

