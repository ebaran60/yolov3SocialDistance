import cv2
import numpy as np

# Calibration needed for each video
def calibrated_dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def isclose(p1, p2):
    c_d = calibrated_dist(p1, p2)
    calib = (p1[1] + p2[1]) / 2
    if 0 < c_d < 0.15 * calib:
        return 1
    else:
        return 0

labelsPath = "./yolo-obj/coco.names"  ## https://github.com/pjreddie/darknet/blob/master/data/coco.names
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = "./yolo-obj/yolov3-tiny.weights" ## https://pjreddie.com/media/files/yolov3-tiny.weights
#https://pjreddie.com/media/files/yolov3.weights
configPath = "./yolo-obj/yolov3-tiny.cfg" ## https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg
#https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
(W, H) = (None, None)

cap = cv2.VideoCapture("VIRAT_S_010204_05_000856_000890.avi")
# the output will be written to output.avi
out = cv2.VideoWriter("output_yolo.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15.,
                      (1080, 720), True)
while True:
    (ret, frame) = cap.read()
    if not ret:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        q = W
    frame = frame[0:H, 200:q]
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if LABELS[classID] == "person":
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
    idNMS = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    if len(idNMS) > 0:
        status = list()
        idflat = idNMS.flatten()
        distance = list()
        center = list()
        val = 0
        for i in idflat:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])
            status.append(0)
        for i in range(len(center)):
            for j in range(len(center)):
                g = isclose(center[i], center[j])
                if g == 1:
                    distance.append([center[i], center[j]])
                    status[i] = 1
                    status[j] = 1
        for i in idflat:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if status[val] == 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 150,0), 2)
            val += 1
        for h in distance:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 150), 2)
    # Write the output
    out.write(frame.astype('uint8'))
    cv2.imshow("preview", frame)
    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
        break
# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window