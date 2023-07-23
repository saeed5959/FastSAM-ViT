"""
    All config is in here
"""
import torch

class ModelConfig:
    """
        All model config
    """

    def __init__(self):
        self.config_path: str = ""
        self.retina_masks: bool = True
        self.imgsz: int = 1024
        self.conf: float = 0.4
        self.iou:float = 0.5
                                                                         


class TrainConfig:
    """
        All train config
    """

    def __init__(self):
        self.save_model: int = 10
        self.epochs: int = 40
        self.batch_size: int = 32
        self.learning_rate: float = 0.00001
        self.step_show: int = 100
        self.device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
