import numpy as np
import torch
from .base import PromptSegAdapter

class SAMMed2DAdapter(PromptSegAdapter):
    def __init__(self, ckpt: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # TODO: SAM-Med2D 的 repo 里通常也会提供 predictor 或与 segment_anything 类似的推理方式
        # 你只要封装成 set_image + predict(boxes)

        self.predictor = None
        raise NotImplementedError("Wire SAM-Med2D loading here.")

    def name(self) -> str:
        return "sam_med2d"

    def set_image(self, rgb_uint8: np.ndarray) -> None:
        raise NotImplementedError

    @torch.inference_mode()
    def predict(self, boxes_xyxy: np.ndarray):
        raise NotImplementedError
