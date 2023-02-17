#参考文献：https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html https://www.researchgate.net/publication/260251570_Automatic_generation_and_detection_of_highly_reliable_fiducial_markers_under_occlusion       
#使用： python test.py --image images/test.png --type DICT_5X5_100
import argparse
import imutils
import cv2
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image containing the ArUCo tag")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Tpe of ArUCo tag to detect")
args = vars(ap.parse_args())
 
#opencv支持的aruco标签
ARUCO_DICT = {"DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
              "DICT_4X4_250": cv2.aruco.DICT_4X4_250, "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
              "DICT_5X5_50": cv2.aruco.DICT_5X5_50, "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
              "DICT_5X5_250": cv2.aruco.DICT_5X5_250, "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
              "DICT_6X6_50": cv2.aruco.DICT_6X6_50, "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
              "DICT_6X6_250": cv2.aruco.DICT_6X6_250, "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
              "DICT_7X7_50": cv2.aruco.DICT_7X7_50, "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
              "DICT_7X7_250": cv2.aruco.DICT_7X7_250, "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
              "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
              "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
              "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
              "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
              "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11}


#载入图片
print("[INFO] Loading image...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
 
#验证aruco标签能否被opencv识别
if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported!".format(args["type"]))
    sys.exit(0)
 
#识别aruco
print("[INFO] Detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
 
if len(corners) > 0:
    ids = ids.flatten()
    for (markerCorner, markerID) in zip(corners, ids):
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))
        cv2.line(image, bottomRight, bottomLeft, (0, 0, 255), 10)
        cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 10)
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
        #arucoID
        #cv2.putText(image, str(markerID), (bottomLeft[0], bottomLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image, 'x', (topLeft[0]-15, topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2)
        cv2.putText(image, 'y', (bottomRight[0]-15, bottomRight[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)
        print("[INFO] ArUco marker ID: {}".format(markerID))
        #结果输出
        cv2.imshow("Image", image)
        cv2.waitKey(0)