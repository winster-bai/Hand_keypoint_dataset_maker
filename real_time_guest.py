from ultralytics import YOLO
import cv2
import math

def vector_2d_angle(v1,v2):
    '''
        求解二维向量的角度
    '''
    v1_x=v1[0]
    v1_y=v1[1]
    v2_x=v2[0]
    v2_y=v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ =65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_

def hand_angle(hand_):
    '''
        获取对应手相关向量的二维角度,根据角度确定手势
    '''
    angle_list = []
    #---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    #---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    #---------------------------- middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    #---------------------------- ring 无名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    #---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

def h_gesture(angle_list):
    '''
        # 二维约束的方法定义手势
        # 只判断手势 0 到 5
    '''
    thr_angle = 65.  # 手指闭合则大于这个值（大拇指除外）
    thr_angle_thumb = 53.  # 大拇指闭合则大于这个值
    thr_angle_s = 49.  # 手指张开则小于这个值
    gesture_str = "Unknown"
    if 65535. not in angle_list:
        print(angle_list)
        if (angle_list[0] > thr_angle_thumb) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
            gesture_str = "0"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
            gesture_str = "1"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
            gesture_str = "2"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (angle_list[3] < thr_angle_s) and (angle_list[4] > thr_angle):
            gesture_str = "3"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s):
            gesture_str = "4"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s):
            gesture_str = "5"
    return gesture_str

def detect():
    model = YOLO("epoch100.pt")  # load a custom model
    cap = cv2.VideoCapture(0)  # 参数0表示调用默认摄像头

    while True:
        ret, frame = cap.read()  # 读取一帧
        if not ret:
            print("无法读取摄像头输入")
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        results = model(frame)  # 将帧输入模型进行推理
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for result in results:
            # 获取图像
            image = result.orig_img  # 这是原始图像

            # 获取关键点
            keypoints = result.keypoints

            # 如果检测到了关键点
            if keypoints is not None:
                # 提取关键点数据
                keypoints_data = keypoints.data[0]  # 获取第一个图像的关键点数据
                hand_points = [(kp[0], kp[1]) for kp in keypoints_data]

                # 检查关键点数量是否足够
                if len(hand_points) >= 21:
                    # 计算手指角度
                    angles = hand_angle(hand_points)

                    # 识别手势
                    gesture = h_gesture(angles)

                    # 在图像上显示手势
                    cv2.putText(image, gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2)

                    for i, kp in enumerate(keypoints_data):
                        x, y, conf = kp
                        # 绘制关键点
                        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
                        # 显示置信度和序号
                        cv2.putText(image, f'{i}', (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 显示检测后的画面
        cv2.imshow('MediaPipe Hands', image)

        # 按Esc键退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect()