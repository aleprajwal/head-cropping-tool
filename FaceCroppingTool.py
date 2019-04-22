'''
print details:
python3 FaceCroppingTool.py -i image_path -v

show orientation and landmarks:
python3 FaceCroppingTool.py -i image_path -o -l
'''

# import necessary libraries
import numpy as np
import argparse
import dlib
import cv2
import os
import imutils
from imutils import face_utils


def landmarkShape(image, rect):
    shape = predictor(image, rect)
    shape = face_utils.shape_to_np(shape)
    return shape


def showLandmarks(shape):
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (255, 0, 0), -1)


# calculate distance
def distance(x, x1, x2):
    d1 = abs(x - x1)  # distance from mid point to left point
    d2 = abs(x - x2)  # distance from mid point to right point
    ratio = d1 / d2
    face_size = [d1, d2]  # width of face  # 1. left to mid and 2. right to mid
    return ratio, face_size


def regionOfInterest(shape, face_bbox, size):  # shape -> face landmarks  # size -> image size
    left = shape[0][0]  # left corner x-axis
    left_new = face_bbox[0]
    right = shape[16][0]  # right corner x-axis
    right_new = face_bbox[2]
    middle_x, middle_y = shape[27]  # middle x and y axis
    chin = shape[8][1]  # chin y-axis

    ratio, face_size = distance(middle_x, left, right)

    if args["verbose"]:
        print("Ratio: ", ratio)
        print("face_size: ", face_size)

    # left faced
    if ratio <= 0.6:
        face_size = face_size[1] * 2
        # x1 = left - int(face_size * 0.2)
        x1 = left_new - int(face_size * 0.1)
        y1 = middle_y - int(face_size * 1.1)
        # x2 = right + int(face_size*0.7)
        x2 = right_new + int(face_size * 0.85)
        y2 = chin + int(face_size * 0.1)
        if args["orientation"]:
            cv2.putText(image, "Left Orientation", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    # right faced
    elif ratio >= 1.6:
        face_size = face_size[0] * 2
        # x1 = left - int(face_size * 0.7)
        x1 = left_new - int(face_size * 0.85)
        y1 = middle_y - int(face_size * 1.1)
        # x2 = right + int(face_size * 0.2)
        x2 = right_new + int(face_size * 0.1)
        y2 = chin + int(face_size * 0.1)
        if args["orientation"]:
            cv2.putText(image, "Right Orientation", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    else:
        face_size = face_size[0] + face_size[1]
        # x1 = left - int(face_size * 0.3)
        x1 = left_new - int(face_size * 0.3)
        y1 = middle_y - int(face_size * 1.1)
        # x2 = right + int(face_size * 0.3)
        x2 = right_new + int(face_size * 0.3)
        y2 = chin + int(face_size * 0.1)
        if args["orientation"]:
            cv2.putText(image, "Straight Orientation", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x2 > size[1]: x2 = size[1]
    if y2 > size[0]: y2 = size[0]

    return x1, y1, x2, y2


# creating Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-v", "--verbose", help="increase output verbosity",
                action="store_true")
ap.add_argument("-o", "--orientation", help="increase output verbosity",
                action="store_true")
ap.add_argument("-l", "--landmark", help="increase output verbosity",
                action="store_true")

# ap.add_argument("-y", "--yolo", required=True,
#                 help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")

args = vars(ap.parse_args())

# initilize dlib face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# initilize haarcascade face detector
face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

# load input image
image = cv2.imread(args["image"])

# check if image is loaded
if image is None:
    print("[Warning] Image not found !")
    exit()

if args["verbose"]:
    print("Original Image shape: {}".format(image.shape))

size = image.shape
# resize image to 640 width and 480 height
# image = imutils.resize(image, width=640, height=480)
# image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
if args["verbose"]:
    print("Resized Image shape: {}".format(image.shape))

# convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detecting face using dlib face detector
if args["verbose"]:
    print("[INFO] Initilizing dlib face detector ......")
face = detector(gray, 1)

# check if face is detected by dlib detector
if not face:
    if args["verbose"]:
        print("### Face not found by dlib face detector"
              "\n[INFO] Initilizing YOLO detector.....")

    # derive the paths to the YOLO weights and model configuration
    # weightsPath = os.path.sep.join([args["yolo"], "yolov3-tiny-face_3000.weights"])
    # configPath = os.path.sep.join([args["yolo"], "yolov3-tiny-face.cfg"])
    weightsPath = "./yolo/yolov3-tiny-face_3000.weights"
    configPath = "./yolo/yolov3-tiny-face.cfg"

    # load custom YOLO object detector trained on face dataset
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and grab its spatial dimensions
    image = cv2.imread(args["image"])
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # crop = image[y:(y + h), x:(x + w)]
            # draw a bounding box rectangle and label on the image
            
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

else:
    if args["verbose"]:
        print("### Face found by dlib face detector")

    for (i, rect) in enumerate(face):
        shape = landmarkShape(gray, rect)
        left = rect.left()
        right = rect.right()
        top = rect.top()
        bottom = rect.bottom()
        if args["verbose"]:
            print("***Dlib rectangle***")
            print("left: ", left)
            print("top: ", top)
            print("right: ", right)
            print("bottom: ", bottom)

        face_bbox = [left, top, right, bottom]

        if args["landmark"]:
            showLandmarks(shape)

        x1, y1, x2, y2 = regionOfInterest(shape, face_bbox, size)
        # crop = image[y1:y2, x1:x2]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 1)
    
if not args["verbose"]:
    cv2.imshow("Bounding Box", image)
    # cv2.imshow("Cropped Image", crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
