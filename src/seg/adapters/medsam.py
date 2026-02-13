import numpy as np
import torch
from .base import PromptSegAdapter

class MedSAMAdapter(PromptSegAdapter):
    def __init__(self, ckpt: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # TODO: MedSAM 常见是 “SAM predictor + 医学权重”
        # 例如使用 segment_anything 的 SamPredictor，再加载 medsam 的权重
        # from segment_anything import sam_model_registry, SamPredictor
        # sam = sam_model_registry["vit_b"](checkpoint=ckpt).to(self.device)
        # self.predictor = SamPredictor(sam)

        self.predictor = None
        raise NotImplementedError("Wire MedSAM loading here.")

    def name(self) -> str:
        return "medsam"

    def set_image(self, rgb_uint8: np.ndarray) -> None:
        # TODO: self.predictor.set_image(rgb_uint8)
        raise NotImplementedError

    @torch.inference_mode()
    def predict(self, boxes_xyxy: np.ndarray):
        # TODO: segment_anything 常见接口:
        # transformed_boxes = self.predictor.transform.apply_boxes_torch(
        #     torch.from_numpy(boxes_xyxy).to(self.device), rgb_uint8.shape[:2]
        # )
        # masks, scores, _ = self.predictor.predict_torch(
        #     point_coords=None,
        #     point_labels=None,
        #     boxes=transformed_boxes,
        #     multimask_output=False,
        # )
        # masks: (N,1,H,W) bool
        # return masks[:,0].cpu().numpy().astype(np.uint8), scores[:,0].cpu().numpy().astype(np.float32)

        raise NotImplementedError
