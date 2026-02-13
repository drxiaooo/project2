from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class PromptSegAdapter(ABC):
    """
    Unified prompt-seg interface:
      - set_image(rgb_uint8): prepare image embeddings
      - predict(boxes_xyxy): return masks (N,H,W) uint8 {0,1} and scores (N,) float (optional)
    """

    @abstractmethod
    def set_image(self, rgb_uint8: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, boxes_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        boxes_xyxy: (N,4) float32 in canvas coords
        returns:
          masks: (N,H,W) uint8
          scores: (N,) float32 (if model doesn't provide, return ones)
        """
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    def close(self) -> None:
        """Optional cleanup."""
        return
