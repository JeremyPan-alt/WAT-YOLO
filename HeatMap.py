import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil
import numpy as np

np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class yolov11_heatmap:
    def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio):
        device = torch.device(device)
        ckpt = torch.load(weight, weights_only=False)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # intersect
        model.load_state_dict(csd, strict=False)  # load
        model.eval()
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')

        target_layers = [eval(layer)]
        method = eval(method)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.__dict__.update(locals())

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
            indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def process_single_image(self, img_path, save_path):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        # img process
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # init ActivationsAndGradients
        grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)

        # get ActivationsAndResult
        result = grads(tensor)
        activations = grads.activations[0].cpu().detach().numpy()

        # postprocess to yolo output
        post_result, pre_post_boxes, post_boxes = self.post_process(result[0])
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf_threshold:
                break

            self.model.zero_grad()
            # get max probability for this prediction
            if self.backward_type == 'class' or self.backward_type == 'all':
                score = post_result[i].max()
                score.backward(retain_graph=True)

            if self.backward_type == 'box' or self.backward_type == 'all':
                for j in range(4):
                    score = pre_post_boxes[i, j]
                    score.backward(retain_graph=True)

            # process heatmap
            if self.backward_type == 'class':
                gradients = grads.gradients[0]
            elif self.backward_type == 'box':
                gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3]
            else:
                gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3] + \
                            grads.gradients[4]
            b, k, u, v = gradients.size()
            weights = self.method.get_cam_weights(self.method, None, None, None, activations,
                                                  gradients.detach().numpy())
            weights = weights.reshape((b, k, 1, 1))
            saliency_map = np.sum(weights * activations, axis=1)
            saliency_map = np.squeeze(np.maximum(saliency_map, 0))
            saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            if (saliency_map_max - saliency_map_min) == 0:
                continue
            saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

            # add heatmap and box to image
            cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)
            "不想在图片中绘画出边界框和置信度，注释下面的一行代码即可"
            '''cam_image = self.draw_detections(post_boxes[i], self.colors[int(post_result[i, :].argmax())],
                                             f'{self.model_names[int(post_result[i, :].argmax())]} {float(post_result[i].max()):.2f}',
                                             cam_image)'''
            # 固定框的颜色是黑色
            cam_image = self.draw_detections(post_boxes[i], self.colors[int(post_result[i, :].argmax()) / 255],
                                             f'{self.model_names[int(post_result[i, :].argmax())]} {float(post_result[i].max()):.2f}',
                                             cam_image)
            cam_image = Image.fromarray(cam_image)
            cam_image.save(f'{save_path}/{i}.png')

    def process_single_image_unified(self, img_path, save_path, draw_boxes=True):
        """
        处理单张图片，将所有检测目标的热力图合并为一张
        """
        # 确保保存路径的文件夹存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 读取和预处理图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            return False

        orig_img = img.copy()  # 保存原始图片
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img_float, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # 初始化梯度计算
        grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)

        # 前向传播
        result = grads(tensor)
        activations = grads.activations[0].cpu().detach().numpy()

        # 后处理获取检测结果
        post_result, pre_post_boxes, post_boxes = self.post_process(result[0])

        # 初始化综合热力图
        combined_saliency_map = None

        # 找到所有高于置信度阈值的检测
        valid_indices = []
        for i in range(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) >= self.conf_threshold:
                valid_indices.append(i)

        if not valid_indices:
            print(f"在图片 {os.path.basename(img_path)} 中没有找到高于阈值的检测")
            return False

        print(f"检测到 {len(valid_indices)} 个目标，生成综合热力图...")

        for idx, i in enumerate(valid_indices):
            self.model.zero_grad()

            # 计算梯度
            if self.backward_type == 'class' or self.backward_type == 'all':
                score = post_result[i].max()
                score.backward(retain_graph=True)

            if self.backward_type == 'box' or self.backward_type == 'all':
                for j in range(4):
                    score = pre_post_boxes[i, j]
                    score.backward(retain_graph=True)

            # 处理热力图
            if self.backward_type == 'class':
                gradients = grads.gradients[0]
            elif self.backward_type == 'box':
                gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3]
            else:
                gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3] + \
                            grads.gradients[4]

            b, k, u, v = gradients.size()
            weights = self.method.get_cam_weights(self.method, None, None, None, activations,
                                                  gradients.detach().numpy())
            weights = weights.reshape((b, k, 1, 1))
            saliency_map = np.sum(weights * activations, axis=1)
            saliency_map = np.squeeze(np.maximum(saliency_map, 0))
            saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))

            # 归一化
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            if (saliency_map_max - saliency_map_min) > 0:
                saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

            # 累积到综合热力图（取最大值）
            if combined_saliency_map is None:
                combined_saliency_map = saliency_map
            else:
                combined_saliency_map = np.maximum(combined_saliency_map, saliency_map)

        # 生成最终的热力图可视化
        if combined_saliency_map is not None:
            # 归一化综合热力图
            combined_saliency_map_min, combined_saliency_map_max = combined_saliency_map.min(), combined_saliency_map.max()
            if (combined_saliency_map_max - combined_saliency_map_min) > 0:
                combined_saliency_map = (combined_saliency_map - combined_saliency_map_min) / (
                            combined_saliency_map_max - combined_saliency_map_min)

            # 将热力图叠加到原图
            heatmap_image = show_cam_on_image(img_float, combined_saliency_map, use_rgb=True)

            # 如果需要，绘制所有边界框
            if draw_boxes:
                for i in valid_indices:
                    '''heatmap_image = self.draw_detections(
                        post_boxes[i],
                        self.colors[(int(post_result[i, :].argmax()))],
                        f'{self.model_names[int(post_result[i, :].argmax())]} {float(post_result[i].max()):.2f}',
                        heatmap_image
                    )'''
                    heatmap_image = self.draw_detections(
                        post_boxes[i],
                        self.colors[0],
                        f'{self.model_names[int(post_result[i, :].argmax())]} {float(post_result[i].max()):.2f}',
                        heatmap_image
                    )

            # 保存结果
            heatmap_image = Image.fromarray(heatmap_image)
            heatmap_image.save(save_path)

            print(f"已保存综合热力图: {save_path}")
            return True

        return False

    def process_folder(self, input_folder, output_folder, unified_heatmap=False):
        """
        批量处理文件夹中的所有图片

        参数:
            input_folder: 输入图片文件夹路径
            output_folder: 输出结果文件夹路径
        """
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(input_folder) if f.lower().endswith(ext)])

        if not image_files:
            print(f"在文件夹 {input_folder} 中没有找到图片文件")
            return

        print(f"找到 {len(image_files)} 张图片，开始处理...")

        # 处理每张图片
        for idx, img_file in enumerate(image_files, 1):
            img_path = os.path.join(input_folder, img_file)

            # 为每张图片创建单独的输出子文件夹
            img_name = os.path.splitext(img_file)[0]

            #是否生成单张综合热力图
            if unified_heatmap:
                # 生成单张综合热力图
                output_path = os.path.join(output_folder, f"{img_name}_unified_heatmap.jpg")
                try:
                    self.process_single_image_unified(img_path, output_path, draw_boxes=True)
                    print(f"  处理完成，结果保存在: {output_folder}")
                except Exception as e:
                    print(f"  处理图片 {img_file} 时发生错误: {e}")
            else:
                # 为每个目标生成单独热力图
                img_output_folder = os.path.join(output_folder, img_name)
                try:
                    # 处理单张图片
                    # self.process_single_image(img_path, img_output_folder)
                    self.process_single_image_unified(img_path, img_output_folder)
                    print(f"  处理完成，结果保存在: {img_output_folder}")
                except Exception as e:
                    print(f"  处理图片 {img_file} 时发生错误: {e}")

            print(f"处理图片 {idx}/{len(image_files)}: {img_file}")

        print(f"\n批量处理完成! 所有结果保存在: {output_folder}")


def get_params():
    params = {
        # 'weight': 'E:/MachineLearning/yolov11-infrared/runs/train/learnable 0.1 0.01/weights/best.pt',  # 训练出来的权重文件
        # 'cfg': 'E:/MachineLearning/yolov11-infrared/ultralytics/cfg/models/11/yolov11-infrared-haar-transformer.yaml',
        'weight': 'E:/MachineLearning/yolov11-infrared/runs/train/final 94.4 0.024 0.25/weights/best.pt',
        'cfg': 'E:/MachineLearning/yolov11-infrared/ultralytics/cfg/models/11/yolov11-infrared-msfa.yaml',
        # 'weight': 'E:/MachineLearning/yolov11-infrared/runs/train/yolov11/weights/best.pt',
        # 'cfg': 'E:/MachineLearning/yolov11-infrared/ultralytics/cfg/models/11/yolo11.yaml',
        # 训练权重对应的yaml配置文件
        'device': 'cuda:0',
        'method': 'GradCAM',         # GradCAMPlusPlus, GradCAM, XGradCAM , 使用的热力图库文件不同的效果不一样可以多尝试
        'layer': 'model.model[8]',   # 想要检测的对应层3, 5, 7, 9
        'backward_type': 'box',      # class, box, all
        'conf_threshold': 0.4,       # 0.6  # 置信度阈值，进度条到一半就停止了就是因为没有高于此值的了
        'ratio': 0.02  # 0.02-0.1
    }
    return params


if __name__ == '__main__':
    # 初始化模型
    model = yolov11_heatmap(**get_params())

    # 设置输入和输出文件夹
    input_folder = "E:/MachineLearning/datasets/YOLO/IR images/valid/images"
    output_folder = "E:/Heatmaps"

    # 批量处理文件夹中的所有图片
    model.process_folder(input_folder, output_folder, unified_heatmap=True)