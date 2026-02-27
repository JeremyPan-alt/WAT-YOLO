import warnings
from ultralytics import YOLO
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # model = YOLO('ultralytics/cfg/models/11/yolov11-infrared-msfa.yaml', verbose=True)
    model = YOLO('ultralytics/cfg/models/v6/yolov6.yaml', verbose=True)
    # model = YOLO('ultralytics/cfg/models/11/yolo11.yaml', verbose=True)
    # model.load('yolov6.pt')
