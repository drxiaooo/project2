import numpy as np
import torch
from .base import PromptSegAdapter

class SAM2Adapter(PromptSegAdapter):
    def __init__(self, ckpt: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # TODO: 按 Meta sam2 官方 repo 接线
        # from sam2.build_sam import build_sam2
        # from sam2.sam2_image_predictor import SAM2ImagePredictor
        # model = build_sam2(cfg, ckpt, device=self.device)
        # self.predictor = SAM2ImagePredictor(model)

        self.predictor = None
        raise NotImplementedError("Wire SAM2 loading here.")

    def name(self) -> str:
        return "sam2"

    def set_image(self, rgb_uint8: np.ndarray) -> None:
        # TODO: self.predictor.set_image(rgb_uint8)
        raise NotImplementedError

    @torch.inference_mode()
    def predict(self, boxes_xyxy: np.ndarray):
        # TODO:
        # masks, scores, _ = self.predictor.predict(
        #     box=boxes_xyxy,
        #     multimask_output=False
        # )
        # return masks.astype(np.uint8), scores.astype(np.float32)
        raise NotImplementedError
