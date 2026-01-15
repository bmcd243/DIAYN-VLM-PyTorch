import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T


class VLMRMPreprocessor:
    """
    Image preprocessing similar to VLM-RM's approach.
    Includes data augmentation for robustness.
    """
    
    def __init__(self, clip_preprocess, augment=True):
        self.clip_preprocess = clip_preprocess
        self.augment = augment
        
        # Additional augmentations for training robustness
        if augment:
            self.augmentation = T.Compose([
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomHorizontalFlip(p=0.3),
            ])
        else:
            self.augmentation = T.Resize(224)
    
    def preprocess_frame(self, frame):
        """
        Preprocess a single frame for CLIP.
        Args:
            frame: numpy array (H, W, 3) in [0, 255]
        Returns:
            preprocessed tensor ready for CLIP
        """
        # Convert to PIL Image
        image = Image.fromarray(frame)
        
        # Apply augmentations if training
        if self.augment:
            image = self.augmentation(image)
        
        # Apply CLIP preprocessing
        return self.clip_preprocess(image)
    
    def preprocess_batch(self, frames):
        """
        Preprocess a batch of frames.
        Args:
            frames: list of numpy arrays or single numpy array
        Returns:
            stacked tensor (B, C, H, W)
        """
        if isinstance(frames, np.ndarray) and len(frames.shape) == 3:
            frames = [frames]
        
        processed = [self.preprocess_frame(f) for f in frames]
        return torch.stack(processed)