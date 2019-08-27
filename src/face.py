import numpy as np
import cv2

img = "C:/Users/Administrator/PycharmProjects/your-eyes/liudehua.jpg"

classfier = cv2.CascadeClassifier('C:/Users/Administrator/PycharmProjects/your-eyes/resources'
                                  '/haarcascade_eye_tree_eyeglasses.xml')


def discern():
    # 读取图片
    frame = cv2.imread(img)
    # 灰度转换
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 人脸检测
    rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(rects) > 0:
        for rect in rects:
            x, y, w, h = rect
            # 打马赛克
            frame[y + 10:y + h - 10, x:x + w, 0] = np.random.normal(size=(h - 20, w))
            frame[y + 10:y + h - 10, x:x + w, 1] = np.random.normal(size=(h - 20, w))
            frame[y + 10:y + h - 10, x:x + w, 2] = np.random.normal(size=(h - 20, w))
        # 保存图片
        cv2.imwrite("liudehua2.jpg",frame)


if __name__ == '__main__':
    discern()
