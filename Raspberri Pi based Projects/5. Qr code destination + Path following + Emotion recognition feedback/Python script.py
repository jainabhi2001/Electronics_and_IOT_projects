import numpy as np
import cv2
from keras.models import model_from_json  
from keras.preprocessing import image  
# import serial

########## THRESHOLD ###########
right_th = 30
left_th = 30

######## Communicating to ARDUINO ########
######## Initialise SERIAL ########
# ser = serial.Serial('/dev/ttyACM0', 9600)

######## Paths for Emotion Recognition MODEL, WEIGHTS, HAARCASCADE ########
model_path = r'C:\Users\Manik\Documents\py\OASIS\face_exp_3\Realtime-emotion-detectionusing-python-master\fer.json'
weight_path = r'C:\Users\Manik\Documents\py\OASIS\face_exp_3\Realtime-emotion-detectionusing-python-master\fer.h5'
cascade_path = r'C:\Users\Manik\Documents\py\OASIS\Face_expression\haarcascade_frontalface_default (1).xml'


######## Load MODEL and WEIGHTS ########
model = model_from_json(open((model_path), "r").read())
model.load_weights(weight_path)  
face_haar_cascade = cv2.CascadeClassifier(cascade_path)

######## List to Store previous Emotions ########
detected_face = []
prev_max_index = -1

######## Initialise the CAMERA to be used ########
cap = cv2.VideoCapture(0)

######## initialize the QRCode detector ########
detector = cv2.QRCodeDetector()

######## BOOLEANS for Different Stages ########
run = False                  # Variable for running cotour detection after successful detection of QR Code 
facebool= False              # This is for detecting face after the robot has stopped
stop = False                 # Stop whole process after Successful FEEDBACK collection

######## YELLOW Contour ########
lower = np.array([7, 170, 147], dtype="uint8")         # Thresholds for the 2 side path lanes -yellow
upper = np.array([28, 232, 231], dtype="uint8")

######## PINK Contour ########
pink_lower = np.array([155,148,168], np.uint8)         # Thresholds for the pink spot to stop
pink_upper = np.array([175,168,248], np.uint8)

def emotionDetector(test_img):
    ######## Detecting Face in GRAY Image ########
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    ######## If FACE is Found ########
    for (x,y,w,h) in faces_detected:  
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
        roi_gray=gray_img[y:y+w,x:x+h]                  # Cropping region of interest i.e. face area from  image  
        roi_gray=cv2.resize(roi_gray,(48,48))           # Resizing the Image   
        img_pixels = image.img_to_array(roi_gray)  
        img_pixels = np.expand_dims(img_pixels, axis = 0)  
        img_pixels /= 255  

        ######## Model Calling and Predicting the Emotion Index ########
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])

        ######## Getting the EMOTION NAME and PRINTING ########
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')  
        predicted_emotion = emotions[max_index]
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  
        
        ######## Tracking the number of CONSECUTIVE SAME EMOTIONS ########
        global prev_max_index
        if max_index == prev_max_index:
            detected_face.append(max_index)
        else:
            detected_face.clear()
            detected_face.append(max_index)
        prev_max_index = max_index

    if(len(detected_face) > 6):
        feedback = detected_face[len(detected_face)-1]
        print("\nFeedback Collected:")
        if  feedback == 3 or feedback == 5:                  # Good
            print("\nExcellent !!\n***\n")
            cv2.putText(test_img, "FeedBack", (int(x+(w/2)), int(y+h)+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (84,96,255), 2)
            cv2.putText(test_img, "  ***", (int(x+(w/2)), int(y+h)+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (84,96,255), 2)
        elif feedback == 6:                                  # Neutral
            print("\nGood !!\n**\n")
            cv2.putText(test_img, "FeedBack", (int(x+(w/2)), int(y+h)+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (84,96,255), 2)
            cv2.putText(test_img, "   **", (int(x+(w/2)), int(y+h)+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (84,96,255), 2)  
        else:                                                # Bad
            print("\nBad !!\n*\n")
            cv2.putText(test_img, "FeedBack", (int(x+(w/2)), int(y+h)+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (84,96,255), 2)
            cv2.putText(test_img, "   *", (int(x+(w/2)), int(y+h)+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (84,96,255), 2)
        cv2.imshow('Feedback Captured',test_img)
        cv2.waitKey(10000)
        return True
    else:
        cv2.imshow('Gathering Feedback',test_img)
        return False

def QR_detector(img):
    data, bbox, _ = detector.detectAndDecode(img)       # Detect and Decode
    if bbox is not None:                                # Check if there is a QRCode in the image
        ele=[]
        ele = data.split(": ")                          # Splitting the data with ": "
        if (ele[0]=="Start") or (ele[0]=="start"):
            print("\nQR code is detected !!\n")
            print(data, "\n")
            return True
    print("Detecting QR !!")
    return False


while True:
    ret,org_image = cap.read()
    org_image = cv2.resize(org_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    original = org_image.copy()

    if not run:
        if not facebool:
            run = QR_detector(org_image)
            cv2.imshow('Scanning QR', org_image)
        else:
            stop = emotionDetector(original)
    
    #```````````````````````````#
    ######## Detect PATH ########
    #...........................#

    elif(run==True):
        ######## BGR image to HSV image ########
        org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2HSV)
        
        ######## YELLOW Mask ########
        mask = cv2.inRange(org_image, lower, upper)                      # MASK for the 2 side path lanes - yellow 
        
        ######## PINK Mask ########
        pink_mask = cv2.inRange(org_image, pink_lower, pink_upper)       # MASK for the pink spot to stop
        kernal = np.ones((5, 5), "uint8")
        pink_mask = cv2.dilate(pink_mask, kernal)

        ######## Get DIMENSIONS of the Image ########
        dimensions = org_image.shape
        width = org_image.shape[1]

        ######## Get Path (YELLOW) Contours ########
        cnts,hierarchy = cv2.findContours(mask, 1, cv2.CHAIN_APPROX_NONE)
        
        ######## Path NOT Detected ########
        if len(cnts) <=0:
            print ("I don't see the line")
        
        ######## Path IS Detected ########
        elif len(cnts) > 0:
            ######## CONTOUR ########
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            ######## Lines through Center ########
            cv2.line(original,(cx,0),(cx,720),(25,56,4),1)
            cv2.line(original,(0,cy),(1280,cy),(25,56,4),1)
            cv2.circle(original, (cx, cy), 5, (48, 39, 210), 2)

            ######## Draw Path CONTOURS ########
            cv2.drawContours(original, cnts, -1, (0,255,0), 1)

            ######## Declaring Thresholds For Turns ########
            left_th = (width // 2) - 20
            right_th = (width // 2) + 20

            ######## Calculate Turns ########
            if cx == 0:
                print ("I don't see the line")
            
            elif cx <= left_th:
                # ser.write(b'l')
                print ("Turn Left!")

            elif cx > left_th and cx < right_th:
                # ser.write(b'f')
                print ("On Track!")

            elif cx >= right_th:
                # ser.write(b'r')
                print ("Turn Right")

            ######## Draw THRESHOLD Lines ########
            cv2.line(original,(left_th,0),(left_th,dimensions[0]),(255,0,125),2)
            cv2.line(original,(right_th,0),(right_th,dimensions[0]),(255,0,125),2)

            ######## Draw PATH Contours ########
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)

        #``````````````````````````````````````#
        ######## Detecting PINK Contour ########
        #......................................#
        contours, hierarchy = cv2.findContours(pink_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            ######## Get AREA of PINK Contour ########
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                original = cv2.rectangle(original, (x, y),(x + w, y + h),(255,20,147), 2)
                cv2.putText(original, "Pink Stop", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(255,20,147))
                print("\nThe bot has stopped, as the destination reached\n")
                print("\nWaiting for a human expression to take feedback")
                facebool=True                       # facebool -> True  : Collect FEEDBACK (EMOTION RECOGNITION)
                run=False                           # run      -> False : STOP ROBOT (PATH RECOGNITION)
                break                               # EXIT Loop
        cv2.imshow('Processing Path', original)

    ######## EXIT [when 'q' is Pressed] (or) [stop == TRUE] ########
    if (cv2.waitKey(10) & 0xFF == ord('q') ) or (stop):
        cap.release()
        cv2.destroyAllWindows()
        # ser.close()
        break
