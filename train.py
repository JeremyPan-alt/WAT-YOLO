import warnings

from PIL.ImageOps import scale

from ultralytics.data.build import seed_worker

from ultralytics import YOLO, RTDETR

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
    # model = YOLO('ultralytics/cfg/models/11/yolov11-infrared-HAEM.yaml')
    # model = YOLO('ultralytics/cfg/models/11/yolov11-infrared-haar-transformer.yaml')
    # model = YOLO('ultralytics/cfg/models/11/yolov11-infrared-msfa.yaml')
    # model = YOLO('ultralytics/cfg/models/11/yolov11-infrared-PPA.yaml')
    # model = YOLO('ultralytics/cfg/models/11/yolov11-infrared-SHSA.yaml')
    # model = YOLO('ultralytics/cfg/models/11/yolov11-infrared-AcMix-wavelet.yaml')
    # model = YOLO('ultralytics/cfg/models/11/yolov11-infrared-haar-wavelet-downsampling.yaml')
    # model = YOLO('ultralytics/cfg/models/11/yolov11-infrared-WTConv.yaml')
    # model = YOLO('ultralytics/cfg/models/11/yolov11-small-object-detection.yaml')
    # model = YOLO('ultralytics/cfg/models/11/yolov11-smallobj-WTConv.yaml')
    # model = YOLO('ultralytics/cfg/models/11/yolov11-infrared-Attn-WTConv.yaml')
    # model = YOLO('ultralytics/cfg/models/11/yolov11-infrared-EUCB-wavelet.yaml')
    model = YOLO('ultralytics/cfg/models/11/WAT-yolo.yaml')
    # model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
    # model = YOLO('ultralytics/cfg/models/v5/yolov5.yaml')
    # model = YOLO('ultralytics/cfg/models/v6/yolov6.yaml')
    # model = YOLO('ultralytics/cfg/models/v8/yolov8-infrared-PPA.yaml')
    # model = YOLO('ultralytics/cfg/models/v8/yolov8-infrared-haarconv.yaml')
    # model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
    # model = YOLO('ultralytics/cfg/models/v8/yolov8-infrared-SHSA-backbone.yaml')
    # model = YOLO('ultralytics/cfg/models/v8/yolov8-infrared-SHSA-neck.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    # model = RTDETR('rtdetr-l.pt')
    path = 'ultralytics/cfg/datasets/Panel_images.yaml'
    # path = 'ultralytics/cfg/datasets/solar-deffects-Dataset.yaml'
    model.train(data=path,
                cache=False,
                pretrained=False,
                warmup_epochs=3,
                # seed=42,
                # deterministic=True,                  # 启用确定性算法，保证重复性，但可能降低性能
                lr0=0.02,
                lrf=0.25,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=4,
                momentum=0.937,
                device='0',
                optimizer='SGD',                       # using SGD
                # patience=0,                          # close earlystop
                # resume=True,
                amp=False,                             # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
