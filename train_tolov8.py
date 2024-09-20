import os
from ultralytics import YOLO

curr_path=os.getcwd()
config_path=os.path.join(curr_path, 'config.yaml')
# config_path

model=YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')

results=model.train(data=config_path, epochs=200, iou=0.5, conf=0.01,lr0=0.01,lrf=0.01,cls=0.5,dfl=1.5,box=8.0,save_period=100)

print("训练完成！请查看文件夹runs/train获取训练结果！")