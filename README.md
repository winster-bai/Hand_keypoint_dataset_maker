# Hand_keypoint_dataset_maker
该仓库使用mediapipe的手势关键点检测功能来采集yolo格式数据
（图片+标签）

## 环境搭建
安装mediapipe
pip install mediapipe
更多详细安装说明见官方库

## 1.mediapipe_collect.py

修改代码中的 image_path 以及 label_path 
运行后弹出摄像头画面，将需要采集的手势放入画面后，按下键盘“s”开始连续采集手势

## 2.yolo模型训练

