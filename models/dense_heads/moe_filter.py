import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)
    
class DeepseekMLP(nn.Module):
    def __init__(self, config, hidden_size = None, intermediate_size = None):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        
        self.input_norm = RMSNorm(self.hidden_size)
        

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False) 
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False) 
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False) 
        self.act_fn = ACT2FN[config.hidden_act] 
        
        import math
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        if self.down_proj.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.down_proj.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.down_proj.bias, -bound, bound)

    def forward(self, x):
        
        x_norm = self.input_norm(x)
        
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
 
            gate = self.act_fn(self.gate_proj(x_norm))
            up = self.up_proj(x_norm)
            down_proj = self.down_proj(gate * up)

        import random
        if self.training and random.random() < 0.001: 
            abs_max = down_proj.abs().max().item()
            weight_max = self.down_proj.weight.abs().max().item()
        return down_proj

class MoEConfig:
    """
    Used to control:
      - Number of experts
      - Number of experts selected per token
      - Gating computation method
      - Auxiliary loss weight
    """

    def __init__(
        self,
        hidden_size=256,            
        moe_intermediate_size=2048,
        n_routed_experts=4,          
        num_experts_per_tok=2,       
        norm_topk_prob=True,        
        scoring_func="softmax",     
        aux_loss_alpha=0.001,       
        seq_aux=False,               
        n_shared_experts=1,
        hidden_act="silu",          
        pretraining_tp=1,
    ):
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.n_shared_experts = n_shared_experts 
        
        self.hidden_act = hidden_act
        self.pretraining_tp = pretraining_tp

class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape        

        hidden_states = hidden_states.reshape(-1, h)  # [batch_size * seq_len, hidden_size]
        
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        import random
        if self.training and random.random() < 0.01: 
            flat_idx = topk_idx.view(-1)
            counts = torch.bincount(flat_idx, minlength=self.n_routed_experts)


        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss
    
    
class DeepseekMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([DeepseekMLP(config, intermediate_size = config.moe_intermediate_size) for i in range(config.n_routed_experts)])
        self.gate = MoEGate(config)
        

        self.gate_norm = RMSNorm(config.hidden_size)
        
        self.num_experts = len(self.experts)

        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekMLP(config=config, intermediate_size = intermediate_size)
            
        self.output_gate = nn.Linear(config.hidden_size, 1)
        self.gate_act = nn.Sigmoid()
        
        nn.init.constant_(self.output_gate.bias, 0.0)
        nn.init.constant_(self.output_gate.weight, 0.0) 

        self.context_layer = nn.MultiheadAttention(
            embed_dim=config.hidden_size, 
            num_heads=4, 
            dropout=0.1,
            batch_first=True
        )
        self.context_norm = RMSNorm(config.hidden_size)
    
    def forward(self, hidden_states, true_counts=None):
        """
        Args:
            hidden_states: [Batch, Seq_Len, Hidden]
            true_counts: [Batch] or None. Records the actual number of tokens per image.
        """

        padding_mask = None
        if true_counts is not None:
            B, L, _ = hidden_states.shape
            device = hidden_states.device
            

            idx_range = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
           
            lens = true_counts.unsqueeze(1)
            

            padding_mask = idx_range >= lens
            
        
        norm_x = self.context_norm(hidden_states)
        

        context_out, _ = self.context_layer(
            query=norm_x, 
            key=norm_x, 
            value=norm_x, 
            key_padding_mask=padding_mask
        )
        

        hidden_states = hidden_states + context_out

        
        identity = hidden_states
        orig_shape = hidden_states.shape

        topk_idx, topk_weight, aux_loss = self.gate(self.gate_norm(hidden_states))
        
        

        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
        
        flat_topk_idx = topk_idx.view(-1)
        if self.training:

            total_tokens = batch_size * seq_len
            hidden_states_repeated = hidden_states_flat.repeat_interleave(self.num_experts_per_tok, dim=0)  # [total_tokens * num_experts_per_tok, hidden_size]
            if hidden_states_repeated.dim() > 2:
                hidden_states_repeated = hidden_states_repeated.reshape(-1, hidden_states_repeated.shape[-1])
            
        
            y = torch.zeros_like(hidden_states_repeated)
            

            for i, expert in enumerate(self.experts):
                
        
                y[flat_topk_idx == i] = expert(hidden_states_repeated[flat_topk_idx == i])
                

            y = y.reshape(-1, self.num_experts_per_tok, hidden_size)
            y = (y * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.reshape(orig_shape)  

            y = AddAuxiliaryLoss.apply(y, aux_loss)

        else:

            flat_topk_idx = topk_idx.view(-1)
            flat_topk_weight = topk_weight.view(-1, 1)

            y = self.moe_infer(hidden_states, flat_topk_idx, flat_topk_weight)

            if y.shape != orig_shape:
         
                y = y.reshape(orig_shape)
            
         
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
               
        
        gate_score = self.gate_act(self.output_gate(y)) # [B, L, 1] (0~1)
        
      
        y_gated = y * gate_score
        
      
        if padding_mask is not None:
      
            y_gated = y_gated.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
        import random
        if self.training and random.random() < 0.01:
            y_abs = y.abs().mean().item()
            gate_val = gate_score.mean().item()
            final_abs = y_gated.abs().mean().item()
            
       
        return y_gated
    

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights, chunk_size=1024):
        """
        x: [batch_size, seq_len, hidden] or [num_tokens, hidden]
        flat_expert_indices: [num_tokens * num_experts_per_tok] 
        flat_expert_weights: [num_tokens * num_experts_per_tok] 
        """
 
        orig_x_shape = x.shape
  
        if x.dim() == 3:
            batch_size, seq_len, hidden_size = x.shape
            x_2d = x.reshape(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
            num_tokens = batch_size * seq_len
        else:
            x_2d = x
            num_tokens = x.shape[0]
            hidden_size = x.shape[1]

        expected_expert_indices = num_tokens * self.num_experts_per_tok
  

      
        if flat_expert_indices.shape[0] != expected_expert_indices:
          
            if flat_expert_indices.shape[0] > expected_expert_indices:
                flat_expert_indices = flat_expert_indices[:expected_expert_indices]
                flat_expert_weights = flat_expert_weights[:expected_expert_indices]
           


        token_positions = torch.arange(num_tokens, device=x.device)
        token_positions = token_positions.repeat_interleave(self.num_experts_per_tok)

   
        expert_cache = torch.zeros_like(x_2d)  # [num_tokens, hidden_size]



        for expert_id in range(self.num_experts):

            mask = flat_expert_indices == expert_id
            expert_mask_sum = mask.sum().item()

            if expert_mask_sum == 0:
                continue

     
            token_pos = token_positions[mask]
            weights = flat_expert_weights[mask]



      
            if token_pos.numel() > 0:
                max_pos = token_pos.max().item()
                min_pos = token_pos.min().item()
        
                if max_pos >= num_tokens:
                
                    valid_mask = token_pos < num_tokens
                    token_pos = token_pos[valid_mask]
                    weights = weights[valid_mask]
     

            if len(token_pos) == 0:

                continue

     
            expert_tokens = x_2d[token_pos]  # [num_tokens_for_expert, hidden_size]
    

            expert_out = self.experts[expert_id](expert_tokens)


            if weights.dim() == 1:
                weights = weights.unsqueeze(-1)  # [N, 1]
            elif weights.dim() == 2 and weights.shape[1] > 1:
                weights = weights[:, :1] 

          
            if weights.shape[1] == 1 and expert_out.shape[1] > 1:
                weights_expanded = weights.expand(-1, expert_out.shape[1])
            else:
                weights_expanded = weights

       
            expert_out_weighted = expert_out * weights_expanded
          
            expert_cache.index_add_(0, token_pos, expert_out_weighted)
      


        if len(orig_x_shape) == 3:
            expert_cache = expert_cache.reshape(orig_x_shape)
         
        return expert_cache
