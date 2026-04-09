import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightKeySelector(nn.Module):
    """
    Principle:
    Combine FPN features and coarse prediction scores to generate a refined importance mask through lightweight convolution.
    """
    def __init__(self, in_channels=256, reduce_channels=64, target_ratio=0.05):
        super().__init__()
        self.target_ratio = target_ratio
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels + 3, reduce_channels, kernel_size=1, bias=False), 
            nn.GroupNorm(num_groups=8, num_channels=reduce_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(reduce_channels, reduce_channels, kernel_size=7, padding=3, groups=reduce_channels, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=reduce_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(reduce_channels, 1, kernel_size=1, bias=True) 
        )
        
        self.adaptive_tau = nn.Sequential(
            nn.Linear(2, 32),           
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()                
        )
        
        self._init_weights()

    def _init_weights(self):
        last = self.adaptive_tau[-2]
        if last.bias is not None:
            nn.init.constant_(last.bias, -5.0)
            
        last_refine = self.refine_conv[-1]
        nn.init.constant_(last_refine.weight, 0)
        if last_refine.bias is not None:
            nn.init.constant_(last_refine.bias, 0)

            
    def forward(self, coarse_score_map, fpn_feats):
        """
        Args:
            coarse_score_map (Tensor): Coarse prediction scores from the original head [B, 1, H, W]
                                     (If the head outputs logits, apply Sigmoid first)
            fpn_feats (Tensor): [New parameter] Corresponding FPN feature map [B, C, H, W]

        Returns:
            importance (Tensor): Refined importance scores [B, 1, H, W]
            tau (Tensor): Dynamic threshold [B, 1]
            mask (Tensor): Binary mask [B, 1, H, W]
        """
        B, _, H, W = coarse_score_map.shape
        
        y_coords = torch.linspace(-1, 1, H, device=fpn_feats.device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(-1, 1, W, device=fpn_feats.device).view(1, 1, 1, W).expand(B, 1, H, W)
        
        cat_feats = torch.cat([fpn_feats, coarse_score_map, x_coords, y_coords], dim=1)
        
        delta = self.refine_conv(cat_feats) # [B, 1, H, W]
        
        importance = torch.sigmoid(coarse_score_map + delta)

        
        score_mean = importance.mean(dim=(2, 3)).view(B, 1)
        score_max = importance.amax(dim=(2, 3)).view(B, 1) 
        
        global_stats = torch.cat([score_mean, score_max], dim=1)
        
        tau = self.adaptive_tau(global_stats) # [B, 1]

        tau_view = tau.view(B, 1, 1, 1)

        mask_hard = (importance > tau_view).float()

        mask = (mask_hard - importance).detach() + importance

        return importance, tau, mask