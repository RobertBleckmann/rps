from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    use_z: bool = False                                            


def landmarks_to_feature_vector(hand_landmarks, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """
    hand_landmarks: list of 21 landmarks, each with .x, .y, .z in normalized coordinates (0..1)
    Output: 1D numpy array (float32)
    """
                    
    if cfg.use_z:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    else:
        pts = np.array([[lm.x, lm.y] for lm in hand_landmarks], dtype=np.float32)

                              
    origin = pts[0].copy()
    pts = pts - origin

                                                                             
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
                                               
        scale = 1.0
    pts = pts / scale

                                 
    return pts.flatten()