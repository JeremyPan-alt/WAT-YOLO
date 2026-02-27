import warnings

from ultralytics.data.build import seed_worker

from ultralytics import YOLO, RTDETR

warnings.filterwarnings('ignore')

# 指定显卡和多卡训练问题 统一都在<YOLOV8V10配置文件.md>下方常见错误和解决方案。
# 训练过程中loss出现nan，可以尝试关闭AMP，就是把下方amp=False的注释去掉。

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolov11-infrared-msfa.yaml')
    #model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
    #model = RTDETR('rtdetr-l.pt')
    path = 'ultralytics/cfg/datasets/Panel_images.yaml'
    # path = 'ultralytics/cfg/datasets/solar-deffects-Dataset.yaml'
    model.predict(data=path,
                  device='0',)
