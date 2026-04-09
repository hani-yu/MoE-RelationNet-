
from typing import Dict, List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from ...core import build_assigner, build_sampler, multi_apply
from ...ops.corner_pool import BRPool, TLPool


from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead

from .moe_filter import DeepseekMoE,MoEConfig
from .LightweightKeySelector import LightweightKeySelector

def reduce_mean(tensor):
    """
    Args:
        tensor:
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


@HEADS.register_module()
class KeypointHead(AnchorFreeHead):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        shared_stacked_convs: int = 0, 
        logits_convs: int = 0, 
        head_types=None, 
        corner_pooling: bool = False, 
        loss_offset=None, 
        **kwargs,
    ) -> None:
        """Predict keypoints of object

        Args:
            num_classes (int): category numbers of objects in dataset.
            in_channels (int): Dimension of input features.
            shared_stacked_convs (int): Number of shared conv layers for all
                keypoint heads.
            logits_convs (int): Number of conv layers for each logits.
            head_types (List[str], optional): Number of head. Each head aims to
                predict different type of keypoints. Defaults to
                ["top_left_corner", "bottom_right_corner", "center"].
            corner_pooling (bool): Whether to use corner pooling for corner
                keypoint prediction. Defaults to False.
            loss_offset (dict, optional): Loss configuration for keypoint offset
                prediction. Defaults to dict(type='SmoothL1Loss',
                loss_weight=1.0/9.0).
            **kwargs:
        """
        if loss_offset is None:
            loss_offset = dict(type="SmoothL1Loss", loss_weight=1.0 / 9.0)
        if head_types is None:
            head_types = ["top_left_corner", "bottom_right_corner", "center"]

        self.corner_pooling = corner_pooling
        self.shared_stacked_convs = shared_stacked_convs
        self.logits_convs = logits_convs
        self.head_types = head_types
        super(KeypointHead, self).__init__(num_classes, in_channels, **kwargs)
        self.loss_offset = build_loss(loss_offset)
    
        self.key_selector = LightweightKeySelector(in_channels=in_channels)
        

        moe_config = MoEConfig(
            hidden_size=256, # GoogLeNet 输出维度
            moe_intermediate_size=512,
            n_routed_experts=4,
            num_experts_per_tok=2,  
            aux_loss_alpha=0.05,    
            norm_topk_prob=True
        )
        self.moe_refiner = DeepseekMoE(config=moe_config)

        for expert in self.moe_refiner.experts:
         
            nn.init.constant_(expert.down_proj.weight, 0)
   
            if hasattr(expert.down_proj, 'bias') and expert.down_proj.bias is not None:
                nn.init.constant_(expert.down_proj.bias, 0)
        
   
        if self.moe_refiner.config.n_shared_experts is not None:
   
            shared_expert = self.moe_refiner.shared_experts
            nn.init.constant_(shared_expert.down_proj.weight, 0)
            if hasattr(shared_expert.down_proj, 'bias') and shared_expert.down_proj.bias is not None:
                nn.init.constant_(shared_expert.down_proj.bias, 0)
                

        if self.train_cfg is not None:
            self.point_assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type="PseudoSampler")
            self.sampler = build_sampler(sampler_cfg, context=self)

    def _init_layers(self) -> None:
        """Construct the model."""
        self.shared_layers = self._init_layer_list(
            self.in_channels, self.shared_stacked_convs
        )

        self.keypoint_layers = nn.ModuleDict() 
        self.keypoint_cls_heads = nn.ModuleDict() 
        self.keypoint_offset_heads = nn.ModuleDict() 

        in_channels = (
            self.in_channels if self.shared_stacked_convs == 0 else self.feat_channels
        )

        for head_type in self.head_types:
            keypoint_layer = self._init_layer_list(in_channels, self.stacked_convs)
            if "corner" in head_type and self.corner_pooling:
                if "top_left" in head_type:
                    keypoint_layer.append(
                        TLPool(
                            self.feat_channels,
                            self.conv_cfg,
                            self.norm_cfg,
                            3,
                            1,
                            corner_dim=64,
                        )
                    )
                else:
                    keypoint_layer.append(
                        BRPool(
                            self.feat_channels,
                            self.conv_cfg,
                            self.norm_cfg,
                            3,
                            1,
                            corner_dim=64,
                        )
                    )
            self.keypoint_layers.update({head_type: keypoint_layer})

            # head
            keypoint_cls_head = self._init_layer_list(
                self.feat_channels, self.logits_convs
            )
            keypoint_cls_head.append(
                nn.Conv2d(self.feat_channels, self.num_classes, 3, stride=1, padding=1)
            )
            self.keypoint_cls_heads.update({head_type: keypoint_cls_head})
     
            keypoint_offset_head = self._init_layer_list(
                self.feat_channels, self.logits_convs
            )

            keypoint_offset_head.append(
                nn.Conv2d(self.feat_channels, 2, 3, stride=1, padding=1)
            )
            self.keypoint_offset_heads.update({head_type: keypoint_offset_head})
  
    def _init_layer_list(self, in_channels: int, num_convs: int) -> nn.ModuleList: 
        """
        Args:
            in_channels (int):
            num_convs (int):
        """
        layers = nn.ModuleList()
        for i in range(num_convs):
            chn = in_channels if i == 0 else self.feat_channels
            layers.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )
        return layers

    def init_weights(self):

        for layer in self.shared_layers:
            normal_init(layer.conv, std=0.01)
            
        for _, layer in self.keypoint_layers.items():
            for m in layer:
                if isinstance(m, ConvModule):
                    normal_init(m.conv, std=0.01)
                 
                else:

                    def _init(m):
                        if isinstance(m, ConvModule):
                            normal_init(m.conv, std=0.01)
                            
                        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                            normal_init(m, std=0.01)
                          
                    m.apply(_init) 
        bias_cls = bias_init_with_prob(0.01)
       
        for head_type, head in self.keypoint_cls_heads.items():
            for i, m in enumerate(head):
                if i != len(head) - 1:
                    normal_init(m.conv, std=0.01)
                else:
              
                    normal_init(m, std=0.01, bias=bias_cls)
        

        for _, head in self.keypoint_offset_heads.items():
            for i, m in enumerate(head):
                if i != len(head) - 1:
                    normal_init(m.conv, std=0.01)
      
                else:
                    normal_init(m, std=0.01)
  

    def forward(
        self, feats: List[torch.Tensor], choices: Union[str, List[str]] = None
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
        """Predict the keypoint and return category and offset.

        Args:
            feats (List[torch.Tensor]): feature map lists. Each is [N,C,Hi,Wi].
            choices (Union[str,List[str]], optional): Select which head to use.

        Returns:
            Tuple[Dict[str,torch.Tensor],Dict[str,torch.Tensor]]: [description]
        """
        B = feats[0].size(0) 
        
        if choices is None:
            choices = self.head_types
        elif isinstance(choices, str):
            choices = [choices]
            
        keypoint_pred = multi_apply(self.forward_single, feats, choices=choices)


        keypoint_scores = keypoint_pred[: len(choices)]
        keypoint_offsets = keypoint_pred[len(choices) :]   
  
        importance_list = []
    
        mask_list = []


        for ch_idx, score_map_list in enumerate(keypoint_scores): 
            for level_idx, score_map in enumerate(score_map_list): 
 
                if isinstance(score_map, (list, tuple)):
                    score_map = score_map[0]  
                    
                if score_map.dim() == 4 and score_map.shape[1] > 1:
    
                    score_map, _ = score_map.max(dim=1, keepdim=True)
                
               
                if not self.training:
                    pad = 1
                    hmax = F.max_pool2d(score_map, (3, 3), stride=1, padding=pad)
                    keep = (hmax == score_map).float()
                    score_map = score_map * keep
             
                
                current_feat = feats[level_idx] 

                importance, tau, mask = self.key_selector(score_map, current_feat)
                
           
                point_ratio = mask.mean()  # scalar in [0,1]
                loss_tau = (point_ratio - self.key_selector.target_ratio) ** 2

             
                if not hasattr(self, "_selector_losses"):
                    self._selector_losses = []
                self._selector_losses.append(loss_tau)

                importance_list.append(importance)
                mask_list.append(mask)
                

        device = feats[0].device
        token_pos_lists = [[] for _ in range(B)]  
        tokens_per_image_by_img = [[] for _ in range(B)]
        

        mask_pairs = []  
        num_levels = len(feats)
        
        for i, mask in enumerate(mask_list):
            level_idx = i % num_levels
            mask_pairs.append((level_idx, mask))


        for pair_idx, (level_idx, mask) in enumerate(mask_pairs):
            raw_feat = feats[level_idx]                    # [B, C, H, W]
            feat_flat = raw_feat.flatten(2).permute(0, 2, 1) # [B, HW, C]
            mask_flat = mask.flatten(2).squeeze(1)         # [B, HW]

            for b in range(B):
                idx = mask_flat[b].nonzero(as_tuple=True)[0]   #
                if idx.numel() > 0:
                    selected = feat_flat[b, idx, :]          
                    tokens_per_image_by_img[b].append(selected)
                    token_pos_lists[b].append((pair_idx, idx))

        final_tokens_list = []
        true_counts_list = []
        for b in range(B):
            if len(tokens_per_image_by_img[b]) == 0:
                
                C = feats[0].shape[1]
                dummy = torch.zeros((1, C), device=device)  
                final_tokens_list.append(dummy)
               
                true_counts_list.append(1) 
            else:
                tokens_cat = torch.cat(tokens_per_image_by_img[b], dim=0)  # [N_b, C]
                final_tokens_list.append(tokens_cat)
                true_counts_list.append(tokens_cat.size(0))

        # pad -> [B, Lmax, C]
        from torch.nn.utils.rnn import pad_sequence
        tokens_padded = pad_sequence(final_tokens_list, batch_first=True)  # [B, Lmax, C]

        true_counts = torch.tensor(true_counts_list, device=device, dtype=torch.long)
        
        moe_out  = self.moe_refiner(tokens_padded, true_counts=true_counts)
   
        if torch.isnan(moe_out).any():
            moe_out = torch.nan_to_num(moe_out, nan=0.0)
            
        moe_out, active_tokens = self.post_filter_tokens(moe_out, energy_threshold=0.15)
   
        moe_feats_per_pair = []  
        Lmax = moe_out.shape[1]

        for seg_i, (level_idx, mask) in enumerate(mask_pairs):
          
            Bf, C, H, W = feats[level_idx].shape
            sparse_map = torch.zeros((Bf, C, H, W), device=device)
            moe_feats_per_pair.append((level_idx, sparse_map))
     
        for b in range(B):
            token_ptr = 0
            pos_segs = token_pos_lists[b]                                                                                        
            
            for (pair_idx, idx) in pos_segs:
                n_pos = idx.numel()
                if n_pos == 0:
                    continue 

                real_n_pos = n_pos
                if token_ptr + n_pos > Lmax:
                    real_n_pos = max(0, Lmax - token_ptr)
                
                if real_n_pos == 0:
                    break

                token_slice = moe_out[b, token_ptr: token_ptr + real_n_pos, :] 

                _, target_sparse_map = moe_feats_per_pair[pair_idx]
                
                current_idx = idx[:real_n_pos]
                
                flatten_size = target_sparse_map.shape[-1] * target_sparse_map.shape[-2]
                if current_idx.max() >= flatten_size:
                    valid_mask = current_idx < flatten_size
                    current_idx = current_idx[valid_mask]
                    token_slice = token_slice[valid_mask]
                
                target_sparse_map.view(B, -1, flatten_size)[b, :, current_idx] = token_slice.transpose(0, 1)

                token_ptr += real_n_pos 
                if real_n_pos < n_pos:
                    break

        moe_feats_list = [torch.zeros_like(feats[lvl]) for lvl in range(num_levels)]
        for (lvl, smap) in moe_feats_per_pair:
            moe_feats_list[lvl] = moe_feats_list[lvl] + smap 
            
        enhanced_feats = []
        for lvl in range(num_levels):
            enhanced_feats.append(feats[lvl] + moe_feats_list[lvl])
  
        keypoint_pred_final = multi_apply(self.forward_single, enhanced_feats, choices=choices)

        num_choices = len(choices)
        keypoint_scores_final_groups = keypoint_pred_final[:num_choices]
        keypoint_offsets_final_groups = keypoint_pred_final[num_choices:]

        final_scores_dict = {}
        final_offsets_dict = {}

        for i, ch in enumerate(choices):
            final_scores_dict[ch] = keypoint_scores_final_groups[i]
            final_offsets_dict[ch] = keypoint_offsets_final_groups[i]

        if self.training:
            return final_scores_dict, final_offsets_dict, enhanced_feats
        else:
            return final_scores_dict, final_offsets_dict, enhanced_feats, importance_list


    def post_filter_tokens(self, moe_out, energy_threshold=0.15):
        """
        Adaptively adjust the threshold based on token count for feature denoising
        After MoE processing, some tokens (keypoints) are enhanced (the magnitude of the feature vector increases), becoming very important;
        While other tokens may be judged by MoE as "this is actually background noise", thus being suppressed (the magnitude of the feature vector becomes very small).
        """
        energy_score = torch.norm(moe_out, dim=-1, keepdim=True)  #
        valid_mask = (energy_score > energy_threshold).float()    

        active_tokens = valid_mask.squeeze(-1).sum(dim=1)       
        return moe_out * valid_mask, active_tokens


    def forward_single(
        self, x: torch.Tensor, choices: List[str]
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): [N,C,H,W]. Input Features.
            choices (List[str]): names of head to use.
        Returns:
            Tuple[torch.Tensor]: head_0_score,...,head_`len(choice)`_score,head_0_offset,...head_`len(choice)`_offset
        """

        feat = x
        for layer in self.shared_layers:
            feat = layer(feat)
        
        keypoint_offsets = []
        keypoint_clses = []


        for head_type in choices:
            keypoint_feat = feat
            for layer in self.keypoint_layers[head_type]:
                keypoint_feat = layer(keypoint_feat)

            offset_feat = cls_feat = keypoint_feat
            for layer in self.keypoint_cls_heads[head_type]:
                cls_feat = layer(cls_feat)
            for layer in self.keypoint_offset_heads[head_type]:
                offset_feat = layer(offset_feat)


            keypoint_clses.append(cls_feat)
            keypoint_offsets.append(offset_feat)
 
        return tuple(keypoint_clses) + tuple(keypoint_offsets)



    def _get_targets_single(
        self,
        gt_points: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        points: torch.Tensor,
        num_points: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute targets for single image.

        Args:
            gt_points (torch.Tensor): Ground truth points for single image with
                shape (num_gts, 2) in [x, y] format.
            gt_bboxes (torch.Tensor): Ground truth bboxes of single image, each
                has shape (num_gt, 4).
            gt_labels (torch.Tensor): Ground truth labels of single image, each
                has shape (num_gt,).
            points (torch.Tensor): Points for all level with shape (num_points,
                3) in [x,y,stride] format.
            num_points (List[int]): Points num for each level.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        assigner = self.point_assigner

        offset_target, score_target, pos_mask = assigner.assign(
            points, num_points, gt_points, gt_bboxes, gt_labels, self.num_classes
        )

        return score_target, offset_target, pos_mask[:, None]

    def get_targets(
        self,
        points: List[torch.Tensor], 
        gt_points_list: List[torch.Tensor],
        gt_bboxes_list: List[torch.Tensor],
        gt_labels_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute regression, classification and centerss targets for points in
        multiple images.

        Args:
            points (List[torch.Tensor]): Points for each level with shape
                (num_points, 3) in [x,y,stride] format.
            gt_points_list ：
            gt_points_list (List[torch.Tensor]): Ground truth points for each
                image with shape (num_gts, 2) in [x, y] format.
            gt_bboxes_list (List[torch.Tensor]): Ground truth bboxes of each
                image, each has shape (num_gt, 4).
            gt_labels_list (List[torch.Tensor]): Ground truth labels of each
                box, each has shape (num_gt,).

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: score targets and offset targets
            and positive_mask for all images, each has shape [batch, num_points,
            channel].
        """
        num_points = [point.size()[0] for point in points]
        points = torch.cat(points, dim=0)
        score_target_list, offset_target_list, pos_mask_list = multi_apply(
            self._get_targets_single,
            gt_points_list,
            gt_bboxes_list,
            gt_labels_list,
            points=points,
            num_points=num_points,
        )

        return (
            torch.stack(score_target_list),
            torch.stack(offset_target_list),
            torch.stack(pos_mask_list),
        )

    @force_fp32(apply_to=("keypoint_scores", "keypoint_offsets"))
    def loss(
        self,
        keypoint_scores: Union[List[torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, torch.Tensor]],
        keypoint_offsets: Union[List[torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, torch.Tensor]],
        keypoint_types: List[str],
        gt_points: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        img_metas: List[dict],
    ) -> Dict[str, torch.Tensor]:

        def unpack_to_list(x):
            """把各种可能的输入结构统一展开成 List[Tensor]"""
            if isinstance(x, torch.Tensor):
                return [x]
            elif isinstance(x, list):
                flat = []
                for item in x:
                    if isinstance(item, list):
                        flat.extend(item)
                    else:
                        flat.append(item)
                return flat
            elif isinstance(x, dict):
                flat = []
                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        flat.append(v)
                    elif isinstance(v, list):
                        flat.extend(v)
                    else:
                        raise TypeError(f"[ERROR]")
                return flat
            else:
                raise TypeError(f"[ERROR] ")

        keypoint_scores = unpack_to_list(keypoint_scores)
        keypoint_offsets = unpack_to_list(keypoint_offsets)

        for i, s in enumerate(keypoint_scores):
            if s.dim() == 3:
                s = s.unsqueeze(0)
                keypoint_scores[i] = s
            elif s.dim() != 4:
                raise ValueError(f"[ERROR]")

        for i, o in enumerate(keypoint_offsets):
            if o.dim() == 3:
                o = o.unsqueeze(1).expand(-1, 2, -1, -1).contiguous()  
                keypoint_offsets[i] = o
            elif o.dim() != 4:
                raise ValueError(f"[ERROR]")

        featmap_sizes = [score.size()[-2:] for score in keypoint_scores]
        points = self.get_points(featmap_sizes, gt_points[0].dtype, gt_points[0].device)

        keypoint_scores = _flatten_concat(keypoint_scores)  # [B, N, C]
        keypoint_offsets = _flatten_concat(keypoint_offsets)  # [B, N, 2]

        score_targets, offset_targets, pos_masks = self.get_targets(
            points, gt_points, gt_bboxes, gt_labels
        )

        keypoint_scores = torch.clamp(torch.nan_to_num(keypoint_scores, nan=0.0, posinf=30.0, neginf=-30.0), -30, 30)
        keypoint_offsets = torch.clamp(torch.nan_to_num(keypoint_offsets, nan=0.0, posinf=1e3, neginf=-1e3), -1e3, 1e3)
        score_targets = torch.nan_to_num(score_targets, nan=0.0)
        offset_targets = torch.nan_to_num(offset_targets, nan=0.0)
        pos_masks = torch.nan_to_num(pos_masks, nan=0.0)

        avg_factor = reduce_mean(torch.sum(pos_masks))
        if avg_factor < 1e-6:
            
            avg_factor = torch.tensor(1.0, device=avg_factor.device)

        loss_cls = self.loss_cls(keypoint_scores, score_targets, avg_factor=avg_factor)
        if not torch.isfinite(loss_cls):
          
            loss_cls = torch.tensor(0.0, device=keypoint_scores.device)

        loss_offset = self.loss_offset(
            keypoint_offsets,
            offset_targets,
            weight=pos_masks.expand_as(keypoint_offsets),
            avg_factor=avg_factor,
        )
        if not torch.isfinite(loss_offset):
            loss_offset = torch.tensor(0.0, device=keypoint_offsets.device)

        losses = {
            "loss_point_cls": loss_cls,
            "loss_point_offset": loss_offset,
        }

        lambda_tau_weight = 0.5   

        if hasattr(self, "_selector_losses") and len(self._selector_losses) > 0:

            loss_tau_total = sum(self._selector_losses)
            
            losses["loss_tau"] = lambda_tau_weight * loss_tau_total

            self._selector_losses = []
 
        return losses



    def loss_multihead(
        self,
        keypoint_scores: Dict[str, List[torch.Tensor]],
        keypoint_offsets: Dict[str, List[torch.Tensor]],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        img_metas: List[dict],
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of multiple heads. :param keypoint_scores: keypoint
        scores for each level for each head. :type keypoint_scores: Dict[str,
        List[torch.Tensor]] :param keypoint_offsets: keypoint offsets for each
        level for each head. :type keypoint_offsets: Dict[str,
        List[torch.Tensor]] :param gt_bboxes: Ground truth bboxes for each image
        with

            shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

        Args:
            keypoint_scores:
            keypoint_offsets:
            gt_bboxes:
            gt_labels (List[torch.Tensor]): class indices corresponding to each
                box.
            img_metas (List[dict]): Meta information of each image, e.g., image
                size, scaling factor, etc.

        Returns:
            Dict[str,torch.Tensor]: Loss for head
        """

        names = list(keypoint_scores.keys())

            
        gt_points = self._box2point(names, gt_bboxes)  
        total_loss_cls = 0
        total_loss_offset = 0
        loss_dict = {}

        for idx, name in enumerate(names):
           
            kp_scores_list = keypoint_scores[name]  
            kp_offsets_list = keypoint_offsets[name]
            gt_pts_list = gt_points[idx]  
            losses = self.loss(
                kp_scores_list,
                kp_offsets_list,
                [name],
                # [gt_points[idx]],
                gt_pts_list,
                gt_bboxes,
                gt_labels,
                img_metas
            )

            total_loss_cls += losses["loss_point_cls"]
            total_loss_offset += losses["loss_point_offset"]

            loss_dict[f"loss_cls_{name}"] = losses["loss_point_cls"]
            loss_dict[f"loss_offset_{name}"] = losses["loss_point_offset"]

 
        loss_dict["stat_cls_total"] = total_loss_cls
        loss_dict["stat_offset_total"] = total_loss_offset
      
        return loss_dict

    def get_keypoints(
        self,
        keypoint_logits: List[torch.Tensor],
        keypoint_offsets: List[torch.Tensor],
        keypoint_score_thr: float = 0.1,
        block_grad: bool = False,
    ) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
    ]:
        """Extract keypoints for single head. Note: For multiple head, we
        propose to concatenate the tensor along batch dimension to speed up this
        process. We do not implement this function for multiple heads as little
        operation is needed for that purpose.
        Args:
            keypoint_scores (List[torch.Tensor]): keypointscores for each level.
            keypoint_offsets (List[torch.Tensor]): keypoint offsets for each level.
            keypoint_features (List[torch.Tensor]): featuremap to select features for each level.
            max_keypoint_num (int): maximum number of selected keypoints. Defaults to 20.
            keypoint_score_thr (float): keypoints with score below this terms are discarded.
        Returns:
            Tuple[List[torch.Tensor],List[torch.Tensor]]: Keypoint scores and positions for each level.
                Each score tensor has shape [batch,max_keypoint_num]. Each
                position tensor has shape [batch,max_keypoint_num,3] in which
                the last dimension indicates [x,y,category].
        """

        featmap_sizes = [hm.size()[-2:] for hm in keypoint_logits]
        points = self.get_points(
            featmap_sizes, keypoint_logits[0].dtype, keypoint_logits[0].device
        )
        keypoint_scores, keypoint_pos, keypoint_inds = multi_apply(
            self.get_keypoints_single,
            keypoint_logits,
            keypoint_offsets,
            points,
            self.strides,
            # max_keypoint_num=max_keypoint_num,
            keypoint_score_thr=keypoint_score_thr,
            block_grad=block_grad,
        )

        return keypoint_scores, keypoint_pos, keypoint_inds, points
       
    def get_keypoints_single(
        self,
        keypoint_logits: torch.Tensor,   # [B, C, H, W]
        keypoint_offsets: torch.Tensor,  # [B, 2, H, W]
        stride: int,
        score_thresh: float = 0.05, 
        max_per_img: int = 100,
        nms_radius: int = 1  
    ):
        """
        Extract keypoints from coarse+MoE refined feature maps.

        Args:
            keypoint_logits (torch.Tensor): [B, C, H, W], after coarse+MoE refinement
            keypoint_offsets (torch.Tensor): [B, 2, H, W], offset predictions
            stride (int): feature map stride

        Returns:
            scores (torch.Tensor): [N], scores of selected keypoints
            keypoint_pos (torch.Tensor): [N,2], x,y coordinates in original image scale
            inds (torch.Tensor): [N,4], indices in (batch, channel, y, x) format
        """
        B, C, H, W = keypoint_logits.shape

        device = keypoint_logits.device

        scores = keypoint_logits.sigmoid()

        kernel_size = nms_radius * 2 + 1 
        padding = nms_radius

        max_mask = F.max_pool2d(scores, kernel_size, stride=1, padding=padding)

        is_peak = (scores == max_mask)

        scores = scores * is_peak.float()
  
        scores_flat = scores.view(B, -1)
        
 
        K = min(max_per_img, scores_flat.shape[1])
        topk_scores, topk_inds = torch.topk(scores_flat, K, dim=1)
        topk_mask = topk_scores > score_thresh # [B, K]
        
     
        valid_counts = topk_mask.sum(dim=1) # [B]

        bad_img_indices = (valid_counts == 0).nonzero(as_tuple=True)[0]
        
   
        if bad_img_indices.numel() > 0:

            safe_k = min(K, 50) 
            topk_mask[bad_img_indices, :safe_k] = True
            
  
            max_val = topk_scores[bad_img_indices[0], 0].item()

        topk_c = (topk_inds // (H * W)).long()
        topk_y = ((topk_inds % (H * W)) // W).long()
        topk_x = ((topk_inds % (H * W)) % W).long()
        

        valid_mask = topk_mask.view(-1) # [B*K]

        points_per_batch = topk_mask.sum(dim=1).tolist()
 
        
        if valid_mask.sum() == 0:
 
            return (
                torch.zeros(0, device=device),
                torch.zeros(0, 2, device=device),
                torch.zeros(0, 4, dtype=torch.long, device=device)
            )
        

        batch_inds = torch.arange(B, device=device).view(B, 1).expand(B, K)
        

        select_b = batch_inds.flatten()[valid_mask]
        select_c = topk_c.flatten()[valid_mask]
        select_y = topk_y.flatten()[valid_mask]
        select_x = topk_x.flatten()[valid_mask]
        
        select_scores = topk_scores.flatten()[valid_mask]
        
        sel_offsets = keypoint_offsets[select_b, :, select_y, select_x] 
        
        keypoint_pos = torch.zeros_like(sel_offsets)
        keypoint_pos[:, 0] = (select_x.float() + sel_offsets[:, 0]) * stride
        keypoint_pos[:, 1] = (select_y.float() + sel_offsets[:, 1]) * stride
        
    
        inds = torch.stack([select_b, select_c, select_y, select_x], dim=1)
        
        return select_scores, keypoint_pos, inds


    def get_keypoints_multihead(
        self,
        keypoint_logits: Dict[str, List[torch.Tensor]],
        keypoint_offsets: Dict[str, List[torch.Tensor]],
        keypoint_choices: List[str],
        map_back: bool = True,
        stride: int = 4, 
    ) -> Tuple[
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
        None,
    ]:
        """
        Extract keypoints for multiple heads using coarse+MoE refinement.

        Args:
            keypoint_logits (Dict[str, List[torch.Tensor]]): dict of [B, C, H, W] per head
            keypoint_offsets (Dict[str, List[torch.Tensor]]): dict of [B, 2, H, W] per head
            keypoint_choices (List[str]): heads to extract
            map_back (bool, optional): whether to split results per head. Defaults to True.
            stride (int, optional): feature map stride. Defaults to 4.

        Returns:
            keypoint_scores: List[List[Tensor]] of keypoint scores per head
            keypoint_pos: List[List[Tensor]] of keypoint positions per head
            keypoint_inds: List[List[Tensor]] of keypoint indices per head
            locations: None (not used in coarse+MoE pipeline)
        """

        keypoint_scores_all = [[] for _ in range(len(keypoint_logits[keypoint_choices[0]]))]
        keypoint_pos_all = [[] for _ in range(len(keypoint_logits[keypoint_choices[0]]))]
        keypoint_inds_all = [[] for _ in range(len(keypoint_logits[keypoint_choices[0]]))]

        for ch in keypoint_choices:

            logits_ch = keypoint_logits[ch]
            offsets_ch = keypoint_offsets[ch]

            if isinstance(logits_ch, torch.Tensor):
                logits_ch = [logits_ch]
                offsets_ch = [offsets_ch]

            for level_idx, (logits, offsets) in enumerate(zip(logits_ch, offsets_ch)):

                scores, pos, inds = self.get_keypoints_single(logits, offsets, stride)

                keypoint_scores_all[level_idx].append(scores)
                keypoint_pos_all[level_idx].append(pos)
                keypoint_inds_all[level_idx].append(inds)

        if map_back:
            return keypoint_scores_all, keypoint_pos_all, keypoint_inds_all, None
        else:

            scores_flat, pos_flat, inds_flat = [], [], []
            for level_idx in range(len(keypoint_scores_all)):

                level_pos = keypoint_pos_all[level_idx]
                level_inds = keypoint_inds_all[level_idx]
                level_scores = keypoint_scores_all[level_idx]


                merged_scores = torch.cat(level_scores, dim=0)  
                merged_pos = torch.cat(level_pos, dim=0)


                merged_inds = torch.cat(level_inds, dim=0)  

                scores_flat.append(merged_scores)  
                pos_flat.append(merged_pos)  
                inds_flat.append(merged_inds) 

            return scores_flat, pos_flat, inds_flat, None


    def get_keypoint_features(
        self,
        feature_sets,
        keypoint_scores,
        keypoint_positions,
        keypoint_inds,
        num_keypoint_head=1,
        selection_method="index",
    ):
        """
        Extract features from sparse keypoints (coarse+MoE pipeline).

        Args:
            feature_sets (List[torch.Tensor]): Feature maps from different levels [B, C, H, W].
            keypoint_scores (List[torch.Tensor]): Sparse keypoint scores [B, K].
            keypoint_positions (List[torch.Tensor]): Sparse keypoint positions [B, K, 2] (x, y).
            keypoint_inds (List[torch.Tensor]): Indices of keypoints in flattened feature maps [B, K].
            num_keypoint_head (int): Number of keypoint heads.
            selection_method (str): "index" or "interpolation" for feature extraction.

        Returns:
            keypoint_features (List[torch.Tensor]): List of feature tensors per level.
            keypoint_positions (List[torch.Tensor]): List of keypoint position tensors per level.
        """

        import torch
        import torch.nn.functional as F
        def debug_list(name, data):
            if data is None:
               
                return
            if not isinstance(data, list):

                return

        debug_list("keypoint_scores", keypoint_scores)
        debug_list("keypoint_positions", keypoint_positions)
        debug_list("keypoint_inds", keypoint_inds)
        

        def unpack_if_tuple_list(lst, name="unknown"):
            if isinstance(lst, list) and len(lst) > 0 and isinstance(lst[0], tuple):
             
                lst = [v for (_, v) in lst]
            return lst

        keypoint_scores = unpack_if_tuple_list(keypoint_scores, "keypoint_scores")
        keypoint_positions = unpack_if_tuple_list(keypoint_positions, "keypoint_positions")
        keypoint_inds = unpack_if_tuple_list(keypoint_inds, "keypoint_inds")

        try:
            B = feature_sets[0].size(0)
            num_levels = len(feature_sets)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Invalid feature_sets: {e}")


        all_keypoint_features = []
        all_keypoint_positions = []


        image_size = list(feature_sets[0].size())[-2:][::-1]
        image_size = [l * self.strides[0] for l in image_size]
 


        def ensure_batch(tensor, B, dim=-1):
            if tensor is None:
             
                return None
            if tensor.dim() == dim:
                tensor = tensor.unsqueeze(0).expand(B, *tensor.shape)
            return tensor

        keypoint_scores = [ensure_batch(s, B) for s in keypoint_scores]
        keypoint_positions = [ensure_batch(p, B, dim=2) for p in keypoint_positions]
        keypoint_inds = [ensure_batch(i, B, dim=2) for i in keypoint_inds]


        def pad_to_levels(lst, num_levels, name="unknown"):
            if len(lst) < num_levels:
             
                B = feature_sets[0].size(0)
                for _ in range(len(lst), num_levels):
                    lst.append(
                        torch.zeros(B, 0, 2, device=device)
                        if name.endswith("positions")
                        else torch.zeros(B, 0, 4, device=device)
                    )
            return lst

        keypoint_positions = pad_to_levels(keypoint_positions, num_levels, "keypoint_positions")
        keypoint_inds = pad_to_levels(keypoint_inds, num_levels, "keypoint_inds")
        keypoint_scores = pad_to_levels(keypoint_scores, num_levels, "keypoint_scores")

        keypoints_per_level = []
        inds_per_level = []

        for lvl in range(num_levels):
            try:
                kp_pos = keypoint_positions[lvl]
                kp_ind = keypoint_inds[lvl]
               
                if isinstance(kp_pos, tuple) and len(kp_pos) == 2:
                    _, kp_pos = kp_pos
                if isinstance(kp_ind, tuple) and len(kp_ind) == 2:
                    _, kp_ind = kp_ind
            
            except IndexError:
                break

            if kp_pos.numel() == 0:
  
                kp_pos = torch.zeros(B, 0, 2, device=feature_sets[lvl].device)
                kp_ind = torch.zeros(B, 0, 4, device=feature_sets[lvl].device)

            keypoints_per_level.append(kp_pos)
            inds_per_level.append(kp_ind)

        def _feature_selection(featuremaps: torch.Tensor, sample_positions: torch.Tensor, sample_inds: torch.Tensor = None):
            """Extract features from featuremaps at sample_positions."""

            B, C, H, W = featuremaps.shape
            N = sample_positions.size(1) if sample_positions is not None else 0
            if N == 0:
                return torch.zeros(B, 0, C, device=featuremaps.device, dtype=featuremaps.dtype)

            featuremaps_flat = featuremaps.reshape(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]

            if selection_method == "index":
                if sample_inds is None:
                    downsample_scale = torch.sqrt((image_size[0] * image_size[1]) / (H * W))
                    sample_inds = (sample_positions[:, :, 1] * W + sample_positions[:, :, 0]) / downsample_scale
                    sample_inds = torch.floor(sample_inds).long()

                else:
                    if sample_inds.dim() == 3 and sample_inds.size(-1) == 4:
                        y_inds = sample_inds[:, :, 2].long()
                        x_inds = sample_inds[:, :, 3].long()
                        sample_inds = y_inds * W + x_inds
                       

                if sample_inds.numel() == 0:
                    return torch.zeros(B, 0, C, device=featuremaps.device, dtype=featuremaps.dtype)

                keypoint_feats = _gather_feat(featuremaps_flat, sample_inds)
               
                return keypoint_feats

            elif selection_method == "interpolation":
                grid = sample_positions * 2.0 / sample_positions.new_tensor(image_size).reshape(1, 1, 2) - 1.0
                result = F.grid_sample(
                    featuremaps, grid.unsqueeze(1), align_corners=False, padding_mode="border"
                ).squeeze(2).permute(0, 2, 1)
              
                return result

            else:
                raise NotImplementedError(f"Selection method {selection_method} not implemented.")

        for lvl, feat_map in enumerate(feature_sets):
          
            try:
                kp_feats_lvl = _feature_selection(feat_map, keypoints_per_level[lvl], inds_per_level[lvl])
                all_keypoint_features.append(kp_feats_lvl)
                all_keypoint_positions.append(keypoints_per_level[lvl])
               
            except Exception as e:
      
                raise

        return all_keypoint_features, all_keypoint_positions



    @staticmethod
    def _box2point(point_types: List[str], boxes: List[torch.Tensor]) -> List[torch.Tensor]:
        """Extract keypoints from bboxes per keypoint type.

        Args:
            point_types (List[str]): keypoint types.
            boxes (List[torch.Tensor]): list of length batch_size, each (num_gts, 4)

        Returns:
            points (List[torch.Tensor]): list of length len(point_types), each (batch_size, num_gts, 2)
        """
        batch_size = len(boxes)
        points = []

        for point_type in point_types:
          
            pts_per_type = []
            for b in boxes:
                if point_type == "top_left_corner":
                    pts_per_type.append(b[:, :2])
                elif point_type == "bottom_right_corner":
                    pts_per_type.append(b[:, 2:])
                elif point_type == "center":
                    pts_per_type.append(b[:, :2]*0.5 + b[:, 2:]*0.5)
                else:
                    raise ValueError(f"Unknown point type: {point_type}")

            points.append(pts_per_type)

        return points


    def get_points(
        self,
        featmap_sizes: List[Tuple[int, int]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (List[Tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            List[torch.Tensor]: points for all levels in each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            y, x = self._get_points_single(
                featmap_sizes[i], self.strides[i], dtype, device, True
            )
            y = y * self.strides[i] + self.strides[i] // 2
            x = x * self.strides[i] + self.strides[i] // 2
            mlvl_points.append(
                torch.stack([x, y, x.new_full(x.size(), self.strides[i])], dim=1)
            )

        return mlvl_points

    def get_bboxes(
        self,
        keypoint_scores: Dict[str, List[torch.Tensor]],
        keypoint_offsets: Dict[str, List[torch.Tensor]],
    ):
        """Get boxes. We will not use this function in our project.

        Args:
            keypoint_scores (Dict[str, List[torch.Tensor]]): keypoint scores for
                each level for each head.
            keypoint_offsets (Dict[str, List[torch.Tensor]]): keypoint offsets
                for each level for each head.
        """
        raise NotImplementedError()


def _concat(
    tensors: Dict[str, List[torch.Tensor]], index: List[str] = None
) -> Tuple[List[str], List[torch.Tensor]]:
    """Concat tensor dict and return their keys, concatenated values.
    Args:
        tensors (Dict[str, List[torch.Tensor]]):
        index (List[str]): Optional.
    """
    if index:
        names = index
    else:
        names = list(tensors.keys())
    return names, [
        torch.cat(values, dim=0) for values in zip(*[tensors[name] for name in names])
    ]


def _split(
    tensors: List[torch.Tensor], keys: List[str]
) -> Dict[str, List[torch.Tensor]]:
    """Rearange tensor list to tensor dict.

    Args:
        tensors (List[torch.Tensor]): [description]
        keys (List[str]): [description]

    Returns:
        Dict[str, List[torch.Tensor]]: [description]
    """
    num_rep = len(keys)
    num_batch = tensors[0].size(0) // num_rep
    return {
        keys[i]: [tensor[num_batch * i : num_batch * (i + 1)] for tensor in tensors]
        for i in range(len(keys))
    }

def _flatten_concat(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    """Flatten multi-level feature maps into [B, sum(H*W), C]."""
    flatten_list = []

    for i, tensor in enumerate(tensor_list):
        if tensor.dim() == 4:
            # [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = tensor.shape
            tensor = tensor.permute(0, 2, 3, 1).reshape(B, -1, C)
         
        elif tensor.dim() == 3:
            B, N, C = tensor.shape
        
        else:
            raise ValueError(f"[ERROR] Unexpected tensor dim={tensor.dim()} at level={i}, shape={tensor.shape}")

        flatten_list.append(tensor)
    out = torch.cat(flatten_list, dim=1)
   
    return out



def _gather_feat(feat: torch.Tensor, ind: torch.Tensor, mask: torch.Tensor = None):
    """Select features with spatial inds.

    Args:
        feat (torch.Tensor): [N,K,C]
        ind (torch.Tensor): [N,M]
        mask (torch.Tensor): [N,M]. Defaults to None.

    Returns:
        feat (torch.Tensor): [N,M,C]
    """

    max_ind = ind.max().item()
    feat_len = feat.size(1)

    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)


    feat = feat.gather(1, ind)

    
    if mask is not None:

        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)

    return feat
