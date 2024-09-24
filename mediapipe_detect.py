# 用于理解mediapipe返回的数据格式和内容

import cv2
import mediapipe as mp

# 初始化手部检测器
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,  # 设置为True以处理静态图像
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils  # 用于绘制关键点的工具

# 读取图像
image = cv2.imread("2.jpg")

# 将图像从BGR转换为RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像提供给手部检测器
results = hands.process(rgb_image)
print("results:")
print("----------------")
print(dir(results))
print("----------------")

# 如果检测到手，绘制关键点
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # 在图像上绘制手部关键点
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 打印关键点的坐标
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            print(f"ID: {id}, X: {cx}, Y: {cy}")

# 显示图像
cv2.imshow("Hand Tracking", image)

# 等待用户按键，按 'q' 键退出
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cv2.destroyAllWindows()
hands.close()