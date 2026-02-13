from .base import PromptSegAdapter
from .medsam2 import MedSAM2Adapter
from .medsam import MedSAMAdapter
from .sam2 import SAM2Adapter
from .sam_med2d import SAMMed2DAdapter

def build_adapter(seg_model: str, ckpt: str, device: str = "cuda") -> PromptSegAdapter:
    seg_model = seg_model.lower()
    if seg_model == "medsam2":
        return MedSAM2Adapter(ckpt, device=device)
    if seg_model == "medsam":
        return MedSAMAdapter(ckpt, device=device)
    if seg_model == "sam2":
        return SAM2Adapter(ckpt, device=device)
    if seg_model in ("sam_med2d", "sam-med2d", "sammed2d"):
        return SAMMed2DAdapter(ckpt, device=device)
    raise ValueError(f"Unknown seg_model: {seg_model}")
