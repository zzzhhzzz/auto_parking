import numpy as np
import time
import cv2

#标定参数

dist=np.array(([[-0.58650416 , 0.59103816, -0.00443272 , 0.00357844 ,-0.27203275]]))
newcameramtx=np.array([[189.076828   ,  0.    ,     361.20126638]
 ,[  0 ,2.01627296e+04 ,4.52759577e+02]
 ,[0, 0, 1]])
mtx=np.array([[398.12724231  , 0.      ,   304.35638757],
 [  0.       ,  345.38259888, 282.49861858],
 [  0.,           0.,           1.        ]])


cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    h1, w1 = frame.shape[:2]
    # 读取摄像头画面
    # 纠正畸变
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h1, w1), 0, (h1, w1))
    dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w1, h1 = roi
    dst1 = dst1[y:y + h1, x:x + w1]
    frame=dst1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters =  cv2.aruco.DetectorParameters_create()
    dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    #检测aruco标记
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

    if ids is not None:

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
        (rvec-tvec).any()

#        aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.1)
#        aruco.drawDetectedMarkers(frame, corners)

        for i in range(rvec.shape[0]):
            cv2.aruco.drawAxis(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
            cv2.aruco.drawDetectedMarkers(frame, corners)
        #ARUCO ID
        cv2.putText(frame, "Id: " + str(ids), (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)


    else:
        cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)


    cv2.imshow("frame",frame)

    key = cv2.waitKey(1)

    if key == 27:         # esc键退出
        print('esc break...')
        cap.release()
        cv2.destroyAllWindows()
        break

    if key == ord(' '):   # 空格键保存
        filename = str(time.time())[:10] + ".jpg"
        cv2.imwrite(filename, frame)
