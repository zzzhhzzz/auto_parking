import cv2
import numpy as np
import glob
 
# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
 
 
def getPoints(data_path, m, n):
    # 获取标定板角点的位置
    objp = np.zeros((m * n, 3), np.float32)
    objp[:, :2] = np.mgrid[0:m, 0:n].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
 
    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点
 
    images = glob.glob(data_path)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (m, n), None)
        # print(ret)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            print(corners2)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.drawChessboardCorners(img, (m, n), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)
 
    print(len(img_points))
    cv2.destroyAllWindows()
    return obj_points, img_points, size
 
 
if __name__ == '__main__':
    obj_points, img_points, size = getPoints("标定测试图片.jpg", 7, 7)
    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    print("ret:", ret)
    print("mtx:\n", mtx)  # 内参数矩阵
    print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
    print("tvecs:\n", tvecs)  # 平移向量  # 外参数
    pass