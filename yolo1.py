import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Load Yolo
net = cv2.dnn.readNet("/home/kasztp/git/yolov3/yolov3.weights", "/home/kasztp/git/yolov3/yolov3.cfg")
classes = []
with open("/home/kasztp/git/yolov3/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
font = cv2.FONT_HERSHEY_PLAIN

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    height,width,channels = frame.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #object detected
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #rectangle co-ordinaters
                x=int(center_x - w/2)
                y=int(center_y - h/2)

                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),1)
                cv2.putText(frame,classes[class_id],(x,y),font,1,(255,255,255),1)           

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
