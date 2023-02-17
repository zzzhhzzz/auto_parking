import numpy as np
import time
import cv2
#读取图片
frame=cv2.imread('./images/2.jpg')
#调整图片大小
#frame=cv2.resize(frame,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_CUBIC)

#灰度话
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#设置预定义的字典
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
#使用默认值初始化检测器参数
parameters =  cv2.aruco.DetectorParameters_create()
#使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray,aruco_dict,parameters=parameters)
#画出标志位置
cv2.aruco.drawDetectedMarkers(frame, corners)
for (markerCorner, markerID) in zip(corners, ids):
    corners = markerCorner.reshape((4, 2))
    (topLeft, topRight, bottomRight, bottomLeft) = corners
    # Convert each of the (x, y)-coordinate pairs to integers
    topRight = (int(topRight[0]), int(topRight[1]))
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    topLeft = (int(topLeft[0]), int(topLeft[1]))

    cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
    cv2.line(frame, bottomRight, bottomLeft, (0, 0, 255), 2)

    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
    cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

    cv2.putText(frame, 'x', (topRight[0]-15, topRight[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
    cv2.putText(frame, 'y', (bottomLeft[0]-15, bottomLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
    
 
cv2.imshow("frame",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()