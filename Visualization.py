import cv2
import os
from ultralytics import solutions
from pathlib import Path
import time


class ImageHeatmapProcessor:
    def __init__(self, input_folder, output_folder, model_path="yolo26n.pt"):
        """
        初始化热力图处理器

        参数:
            input_folder: 输入图片文件夹路径
            output_folder: 输出图片文件夹路径
            model_path: YOLO模型路径或名称
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.model_path = model_path

        # 创建输出文件夹
        os.makedirs(self.output_folder, exist_ok=True)

        # 支持的图片格式
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    def get_image_files(self):
        """获取输入文件夹中的所有图片文件"""
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(self.input_folder.glob(f'*{ext}'))
            image_files.extend(self.input_folder.glob(f'*{ext.upper()}'))
        return sorted(image_files)  # 按文件名排序

    def process_images(self, show_progress=True, save_format='jpg', region_points=None, classes=None):
        """
        处理所有图片并生成热力图

        参数:
            show_progress: 是否显示进度信息
            save_format: 保存的图片格式 ('jpg', 'png', 'bmp')
            region_points: 可选，区域点列表
            classes: 可选，要处理的类别列表
        """
        # 获取所有图片文件
        image_files = self.get_image_files()

        if not image_files:
            print(f"在文件夹 {self.input_folder} 中没有找到图片文件")
            return 0

        print(f"找到 {len(image_files)} 张图片")

        # 初始化heatmap对象
        heatmap = solutions.Heatmap(
            show=False,  # 批量处理时不显示窗口
            model=self.model_path,
            colormap=cv2.COLORMAP_PARULA,
            region=region_points,
            classes=classes,
            conf=0.4,
        )

        processed_count = 0
        start_time = time.time()

        # 处理每张图片
        for idx, img_path in enumerate(image_files, 1):
            if show_progress:
                print(f"处理中: {idx}/{len(image_files)} - {img_path.name}")

            # 读取图片
            im0 = cv2.imread(str(img_path))

            if im0 is None:
                print(f"  警告: 无法读取图片 {img_path}")
                continue

            try:
                # 生成热力图
                results = heatmap(im0)

                # 获取处理后的图像
                result_image = results.plot_im

                # 构建输出文件路径
                output_filename = f"{img_path.stem}_heatmap.{save_format}"
                output_path = self.output_folder / output_filename

                # 保存结果图片
                save_params = []
                if save_format.lower() == 'jpg' or save_format.lower() == 'jpeg':
                    save_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

                success = cv2.imwrite(str(output_path), result_image, save_params)

                if success:
                    processed_count += 1
                    if show_progress:
                        print(f"  已保存: {output_filename}")
                else:
                    print(f"  错误: 保存失败 {output_filename}")

            except Exception as e:
                print(f"  错误: 处理图片 {img_path.name} 时发生异常: {e}")
                continue

        # 计算处理时间
        elapsed_time = time.time() - start_time

        print(f"\n处理完成!")
        print(f"成功处理: {processed_count}/{len(image_files)} 张图片")
        print(f"耗时: {elapsed_time:.2f} 秒")
        print(f"平均每张: {elapsed_time / max(processed_count, 1):.2f} 秒")
        print(f"输出目录: {self.output_folder}")

        return processed_count


# 使用示例
if __name__ == "__main__":
    # 配置参数
    INPUT_FOLDER = "E:/MachineLearning/datasets/YOLO/IR images/valid/images"  # 替换为你的输入文件夹路径
    OUTPUT_FOLDER = "E:/Heatmaps"  # 替换为你的输出文件夹路径
    # MODEL_PATH = "E:/MachineLearning/yolov11-infrared/runs/train/yolov11/weights/best.pt"  # 替换为你的模型路径
    MODEL_PATH = "E:/MachineLearning/yolov11-infrared/runs/train/learnable 0.1 0.01/weights/best.pt"  # 替换为你的模型路径

    # 可选：定义检测区域（多边形）
    # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # 矩形区域

    # 可选：指定要检测的类别（例如：0=person, 2=car）
    # classes = [0, 2]

    # 创建处理器实例
    processor = ImageHeatmapProcessor(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        model_path=MODEL_PATH
    )

    # 处理图片
    processor.process_images(
        show_progress=True,
        save_format='jpg',  # 可选: 'png', 'jpg', 'bmp'
        # region_points=region_points,  # 可选
        # classes=classes  # 可选
    )