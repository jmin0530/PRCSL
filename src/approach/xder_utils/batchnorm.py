import torch
import torch.nn as nn

class bn_track_stats:
    def __init__(self, module: nn.Module, condition=True):
        """
        Context manager for temporarily disabling batch normalization tracking statistics.

        Args:
            module (nn.Module): The module containing batch normalization layers.
            condition (bool, optional): Whether to enable or disable tracking statistics. Defaults to True.
        """
        self.module = module
        self.enable = condition

    def __enter__(self):
        """
        Disable batch normalization tracking statistics if the condition is False.
        """
        if not self.enable:
            for m in self.module.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.track_running_stats = False

    def __exit__(self, type, value, traceback):
        """
        Enable batch normalization tracking statistics if the condition is False.
        """
        if not self.enable:
            for m in self.module.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.track_running_stats = True