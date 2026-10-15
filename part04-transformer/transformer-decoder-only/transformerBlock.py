import math
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        #三套独立的可学习投影
        self.q_proj = nn.Linear(d_model, d_model, bias = False)
        self.k_proj = nn.Linear(d_model, d_model, bias = False)
        self.v_proj = nn.Linear(d_model, d_model, bias = False)
        
        #多个head拼接后再一次进行特征混合
        self.out_proj = nn.Linear(d_model, d_model, bias = False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        #buffer不参与训练，但会随模型移动到CPU/GPU
        #也会保存在state_dict中
        causal_mask = torch.tril(
            torch.ones(
                max_seq_len,
                max_seq_len,
                dtype = torch.bool,
            )
        )
        
        self.register_buffer(
            "causal_mask",
            causal_mask.view(1, 1, max_seq_len, max_seq_len),
        )
        
    def split_heads(self, x):
        B, T, D = x.shape
        
        # [B, T, D]
        # -> [B, T, H, head_dim]
        # -> [B, H, T, head_dim]
        
        return (
            x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) #交换第一维，第二维
        )
    
    def merge_heads(self, x):
        B, H, T, head_dim = x.shape
        
        #transpose后内存不连续
        #contiguous() 先整理内存，再用view合并维度
        
        return(
            x.transpose(1, 2)
            .contiguous()
            .view(B, T, H * head_dim)
        )
    
    def forward(self, x, past_kv = None, use_cache = False,):
        B, T, D = x.shape
        
        q = self.split_heads(self.q_proj(x))
        k_new = self.split_heads(self.k_proj(x))
        v_new = self.split_heads(self.v_proj(x))
        
        if past_kv is None:
            past_length = 0
            k = k_new
            v = v_new
        else:
            past_k, past_v = past_kv
            past_length = past_k.shape[-2]
            
            #沿序列维追加新的token
            k = torch.cat([past_k, k_new], dim = -2)
            v = torch.cat([past_v, v_new], dim = -2)
        
        total_length = k.shape[-2]
        
        if total_length > self.causal_mask.shape[-1]:
            raise ValueError(
                f"KV cache length {total_length} exceeds "
                f"max_seq_len {self.causal_mask.shape[-1]}"
            )
            
        #每个head单独计算attention
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        
        #当前query对应的绝对位置范围是past_length ... past_length + T -1
        mask = self.causal_mask[
            :,
            :,
            past_length : past_length + T, 
            :total_length,
        ]
        
        #mask=False的未来位置变成 -inf
        scores = scores.masked_fill(~mask, float("-inf"))
        
        weights = torch.softmax(scores, dim = -1)
        weights = self.attn_dropout(weights)
        
        context = weights @ v
        context = self.merge_heads(context)
        
        output = self.out_proj(context)
        output = self.out_dropout(output)
        
        if use_cache:
            present_kv = (k, v)
            return output, present_kv
        
        return output
    
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-5):
        super().__init__()
        
        #nn.Parameter: 注册为可学习参数，优化器会更新它
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        # x:[B, T, D]
        # keepdim = True 保留最后一维，方便广播
        mean = x.mean(dim = -1, keepdim = True)
        
        #LayerNorm使用除以D的方差，不用D-1
        variance = (x - mean).square().mean(
            dim = -1, keepdim = True
        )
        
        normalized = (x - mean) / torch.sqrt(
            variance + self.eps
        )
        
        return self.gamma * normalized + self.beta

class FeedForward(nn.Module):
    def __init__(self, d_model, expansion = 4, dropout = 0.0):
        super().__init__()
        
        hidden_dim = expansion * d_model
        
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        max_seq_len,
        dropout = 0.0,
    ):
        super().__init__()
        
        #两个子层分别使用自己的LayerNorm参数
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        
        self.attention = CausalSelfAttention(
            d_model = d_model,
            num_heads = num_heads,
            max_seq_len = max_seq_len,
            dropout = dropout,
        )
        
        self.ffn = FeedForward(
            d_model = d_model,
            dropout = dropout,
        )
    
    def forward(
        self,
        x,
        past_kv = None,
        use_cache = False,
    ):
        normalized = self.ln1(x)
        
        if use_cache:
            attention_output, present_kv = self.attention(
                normalized,
                past_kv = past_kv,
                use_cache = True
            )
        else:
            attention_output = self.attention(normalized)
            
        x = x + attention_output
        x = x + self.ffn(self.ln2(x))
        
        if use_cache:
            return x, present_kv
        
        return x