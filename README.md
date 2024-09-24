# Hand_keypoint_dataset_maker
该仓库使用mediapipe的手势关键点检测功能来采集yolo格式数据
（图片+标签）

## 环境搭建
安装mediapipe
pip install mediapipe
更多详细安装说明见官方库
安装完成后可以运行 mediapipe_detect.py 测试是否安装完成

## 1.mediapipe_collect.py

修改代码中的 image_path 以及 label_path 
运行后弹出摄像头画面，将需要采集的手势放入画面后，按下键盘“s”开始连续采集手势数据
采集到的图片和标签名一一对应，标签中格式为{0 中心坐标X 中心坐标Y 宽 高 21个关键点的XY坐标}
其中坐标值通过读取mediapipe输出信息中的hand_landmarks.landmar获取，并除以图像画面宽高来获得比例信息

## 2.yolo模型训练
1.修改config.yaml中的图像和标签路径，使其与采集的图像标签数据集路径一致
2.运行train_yolov8.py （根据需要修改训练参数）

## 3.实时检测
修改real_time_guest.py中的path_to_your_model.pt为你的模型路径（通常保存在runs/train/weights文件夹下），可按照个人需求修改h_gesture()函数中的类别来检测更多手势


## 4.其他
如果识别结果差,检查实时识别的手势标签是否符合mediapipe标准标注说明（见下图）
![图片](image/hand_landmarks.png)


如果关键点绑定效果好但是序号错乱，可以直接修改real_time_guest.py中hand_angle()函数的序号定义来实现手势区分