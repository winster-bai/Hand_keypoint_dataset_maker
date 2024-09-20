import cv2
import mediapipe as mp
import os

# 初始化手检测器
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# 创建保存图像和标签的文件夹
os.makedirs('add_dataset1/image/val', exist_ok=True) 
os.makedirs('add_dataset1/label/val', exist_ok=True)

# 读取摄像头图像
cap = cv2.VideoCapture(0)
# 初始化计数器
counter = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img_height, img_width, _ = img.shape
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)

    # 初始化 img_show 变量
    img_show = img.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 获取所有关键点的坐标
            h, w, c = img.shape
            x_list = []
            y_list = []

            for id, lm in enumerate(hand_landmarks.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                x_list.append(px)
                y_list.append(py)

                # 在图像上显示关键点序号
                cv2.putText(img_show, str(id), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # 计算边界框
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            # 计算中心点和宽高
            x_center = (x_min + x_max) / 2 / w
            y_center = (y_min + y_max) / 2 / h
            width = (x_max - x_min) / w
            height = (y_max - y_min) / h

            # 数据转化成画面比例
            landmarks_with_ratio_and_2 = [f'{lm.x} {lm.y} 2' for lm in hand_landmarks.landmark]

            # 打印边界框信息和关键点
            # print(f'Center: ({x_center}, {y_center}), Width: {width}, Height: {height}')
            # print('Landmarks:', ' '.join(landmarks_with_ratio_and_2))

            # 可视化关键点
            mp_draw.draw_landmarks(img_show, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 显示图像
    cv2.imshow('Image', img_show)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        # 保存图像
        image_path = f'add_dataset1/image/val/{counter}.jpg'
        cv2.imwrite(image_path, img)

        # 保存标签
        label_path = f'add_dataset1/label/val/{counter}.txt'
        with open(label_path, 'w') as f:
            f.write(f'0 {x_center} {y_center} {width} {height} {" ".join(landmarks_with_ratio_and_2)}\n')
        print(f'Saved {image_path} and {label_path}')

        # 增加计数器
        counter += 1

cap.release()
cv2.destroyAllWindows()