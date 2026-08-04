import json
import os
import sys
sys.path.append('/data/wcf/ascent/')
from third_party.LINEA.models.linea.hgnetv2 import build_hgnetv2
from third_party.LINEA.models.linea.hybrid_encoder import build_hybrid_encoder
from third_party.LINEA.models.linea.decoder import build_decoder
from third_party.LINEA.models.linea.linea import LINEA, PostProcess
from model_api.server_wrapper_out import ServerMixin, host_model, send_request, str_to_image

def build_model(args):
    backbone = build_hgnetv2(args)
    encoder = build_hybrid_encoder(args)
    decoder = build_decoder(args)
    model = LINEA(backbone, encoder, decoder)
    post_process = PostProcess()
    return model, post_process


LINEA_CONFIG = "third_party/LINEA/configs/linea/linea_hgnetv2_l.py"
LINEA_WEIGHTS = "pretrained_weights/linea_hgnetv2_l.pth"

from third_party.LINEA.util.slconfig import DictAction, SLConfig
import numpy as np
import torch
from torch import nn
import torchvision.transforms as T
from PIL import Image
#download_url = f"https://github.com/SebastianJanampa/storage/releases/download/LINEA/linea_hgnetv2_{model_size}.pth"
#state = torch.hub.load_state_dict_from_url(download_url, map_location=torch.device('cuda'), model_dir=weights_folder)


import numpy as np

def get_line_params(line):
    """计算线段的极坐标参数 (rho, theta)"""
    x1, y1, x2, y2 = line
    angle = np.arctan2(y2 - y1, x2 - x1)
    # 将角度归一化到 [0, pi) 范围内，因为平行线方向相同或相反
    theta = angle % np.pi

    # 计算原点到直线的距离 rho
    # 极坐标方程: rho = x*cos(theta) + y*sin(theta)
    # 注意：这里的 theta 是法线角度，所以需要根据斜率角度转换
    normal_theta = (theta + np.pi / 2) % np.pi
    rho = x1 * np.cos(normal_theta) + y1 * np.sin(normal_theta)
    return rho, normal_theta
    #print(line)

def extract_parallel_lines_ransac(lines, angle_threshold=np.deg2rad(10.0), iterations=1000):
    """
    使用 RANSAC 逻辑提取具有相同主方向的平行线
    lines: 形状为 (N, 4) 的 numpy 数组
    """
    best_inliers = []
    best_theta = 0

    # 预计算所有线段的 rho 和 theta
    #print(lines.shape)
    params = np.array([get_line_params(l) for _, l in enumerate(lines)])
    rhos = params[:, 0]
    thetas = params[:, 1]

    for _ in range(iterations):
        # 1. 随机采样一条线段作为“模型”
        sample_idx = np.random.randint(len(lines))
        sample_theta = thetas[sample_idx]
        if abs(sample_theta - np.pi/2) > np.deg2rad(40): # 偏离水平超过40度则不采样
          #print(sample_idx)
          continue
        # 2. 计算所有线段与采样线段的角度差
        # 处理角度的周期性 (0 和 pi 是同一个方向)
        diff = np.abs(thetas - sample_theta)
        diff = np.minimum(diff, np.pi - diff)

        # 3. 找出所有角度接近的线段（即潜在的平行线）
        inliers_mask = diff < angle_threshold
        num_inliers = np.sum(inliers_mask)

        if num_inliers > len(best_inliers):
            best_inliers = np.where(inliers_mask)[0]
            best_theta = sample_theta
    if len(best_inliers) == 0:
        print("No parallel lines found.")
        return [], 0
    # ext_lines = lines[best_inliers]
    # mid_x = (ext_lines[:, 0] + ext_lines[:, 2]) // 2
    # img_width = 640 # 假设输入图像宽度为640
    # mid_x = mid_x - img_width / 2 # 转换为以图像
    # direction = 0
    # if len(mid_x[mid_x>0])>len(mid_x[mid_x<0]):
    #     direction = 3 # 主方向在右侧
    # elif len(mid_x[mid_x>0])<len(mid_x[mid_x<0]):
    #     direction = 2 # 主方向在左侧
    #print(direction)
    # 2. 寻找 rho 值最接近中位数的线段索引
    # idx 是 best_inliers 数组中的位置
    #idx_in_inliers = np.argmin(np.abs(rhos[best_inliers] - median_rho))


    # 返回主方向上的所有平行线段及其对应的 rho 值
    return best_inliers, 0

class LineaModel:
    def __init__(
        self,
        config_path: str = LINEA_CONFIG,
        weights_path: str = LINEA_WEIGHTS,
        thrh: float = 0.2,
        device: torch.device = torch.device("cuda"),
    ):
        cfg = SLConfig.fromfile(config_path)
        cfg.pretrained = False
        model, postprocessor = build_model(cfg)
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=True)
        self.model = model.deploy()
        self.model.to(device)
        self.postprocessor = postprocessor.deploy()
        self.device = device
        self.thrh = thrh

    def predict(self, image: np.ndarray) -> tuple[int, list[list[int, int]]]:
        """
        This function makes predictions on an input image tensor or numpy array using a
        pretrained model.

        Arguments:
            image (np.ndarray): An image in the form of a numpy array.

        Returns:
            np.ndarray: An array containing the predicted lines.
        """
        h, w, _ = image.shape
        orig_size = torch.tensor([[w, h]]).to(self.device)
        transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
            T.Normalize(mean=[0.538, 0.494, 0.453], std=[0.257, 0.263, 0.273]),
        ])     # 将图像转换为张量并进行预处理
        im_data = transforms(Image.fromarray(image)).unsqueeze(0).to(self.device)  # 添加批次维度并移动到设备
        output = self.model(im_data)
        output = self.postprocessor(output, orig_size)
        lines, scores = output
        scr = scores[0]
        line = lines[0][scr > self.thrh]
        temp_lines = line.cpu().numpy()
        if temp_lines.shape[0] == 0:
            return (0, [])
        #scrs = scr[scr > self.thrh]
        #print(f"lines shape: {lines.shape}, scores shape: {scores.shape}")
        dx = temp_lines[:, 2] - temp_lines[:, 0]
        dy = temp_lines[:, 3] - temp_lines[:, 1]
        lengths = np.sqrt(dx**2 + dy**2)

        # 2. 仅保留长度大于阈值的线段
        mask = lengths > 200
        temp_lines = temp_lines[mask]
        if temp_lines.shape[0] == 0:
            return (0, [])

        parallel_lines, _ = extract_parallel_lines_ransac(temp_lines)
        # line = line[parallel_lines]
        # scrs = scrs[parallel_lines]
        print(f"Detected {len(parallel_lines)} parallel lines,{temp_lines[parallel_lines]}")
        if(len(parallel_lines) == 0):
            return (0, [])
        mid_point_list = []
        for line in parallel_lines:
            mid_point_list.append([(temp_lines[line][0] + temp_lines[line][2]) // 2, (temp_lines[line][1] + temp_lines[line][3]) // 2])
        mid_points = np.array(mid_point_list)
        mid_points[:, 0] = np.clip(mid_points[:, 0], 0, w - 1)
        mid_points[:, 1] = np.clip(mid_points[:, 1], 0, h - 1)
        return (int(len(mid_point_list)), mid_points.tolist()) # 返回平行线数量和主方向线段的平均列坐标与图像中心的距离


class LineaModelClient:
    def __init__(self, port: int = 13187):
        host = os.getenv("OUT_HOST", "localhost")
        self.url = f"http://{host}:{port}/linea"

    def predict(self, image_numpy: np.ndarray) -> tuple[int, np.ndarray]:
        try:
            response = send_request(self.url, image=image_numpy)
            num_lines, col = response["num_lines"], response["col"]
            #out = np.array(out)
            return (int(num_lines), np.array(col))
        except Exception as e:
            print(f"Request failed: {e}")
            return (0, np.array([]))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=13187)
    args = parser.parse_args()

    print("Loading model...")

    class LineaServer(ServerMixin, LineaModel):
        def process_payload(self, payload: dict) -> dict:
            # 处理 GET 请求
            if payload is None:
                return {"status": "ok", "message": "Linea service is running"}

            # 处理 POST 请求
            image = str_to_image(payload["image"])
            num_lines, col = self.predict(image)
            return {"num_lines": num_lines, "col": col}

    gdino = LineaServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(gdino, name="linea", port=args.port)