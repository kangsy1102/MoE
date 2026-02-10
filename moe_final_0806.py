
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. Attention Pooling ---

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, attention_dim=None):
        super().__init__()
        if attention_dim is None:
            attention_dim = input_dim
            
        self.query_projection = nn.Linear(input_dim, attention_dim)  # Wq
        self.key_projection = nn.Linear(input_dim, attention_dim)    # Wk
        self.value_projection = nn.Linear(input_dim, attention_dim)  # Optional: Wv

        self.global_context = nn.Parameter(torch.empty(input_dim))   # Raw vector
        nn.init.xavier_uniform_(self.global_context.unsqueeze(0))

        self.layer_norm = nn.LayerNorm(attention_dim)

    def forward(self, x, mask=None):
        keys = self.key_projection(x)
        values = self.value_projection(x)

        global_query = self.global_context.unsqueeze(0).expand(x.size(0), -1)
        query = self.query_projection(global_query)

        attention_scores = torch.matmul(query.unsqueeze(1), keys.transpose(-2, -1)).squeeze(1)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        pooled_output = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)

        return self.layer_norm(pooled_output), attention_weights


# --- 2. Expert ---
class Expert(nn.Module):
    def __init__(self, expert_input_dim, expert_output_dim, hidden_dim1, hidden_dim2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(expert_input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, expert_output_dim)
        )
    def forward(self, x_expert_input):
        return self.network(x_expert_input)

# --- 3. MoE Router ---
class MoERouter(nn.Module):
    def __init__(self, router_input_dim, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.gate_linear = nn.Linear(router_input_dim, num_experts)
        self.expert_biases = nn.Parameter(torch.zeros(num_experts))

    def forward(self, x_router_input):
        initial_affinity_scores = self.gate_linear(x_router_input)
        scores_for_selection = initial_affinity_scores + self.expert_biases
        _, top_k_indices = torch.topk(scores_for_selection, self.top_k, dim=-1)
        gating_weights_raw = initial_affinity_scores.gather(dim=-1, index=top_k_indices)
        return F.softmax(gating_weights_raw, dim=-1), top_k_indices

# --- 4. MoE Layer ---
class MoELayer(nn.Module):
    def __init__(self, router_input_dim, expert_input_dim, expert_output_dim, num_experts, top_k, expert_hidden_dims):
        super().__init__()
        self.router = MoERouter(router_input_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(expert_input_dim, expert_output_dim,
                   hidden_dim1=expert_hidden_dims[0],
                   hidden_dim2=expert_hidden_dims[1])
            for _ in range(num_experts)
        ])
        self.expert_output_dim = expert_output_dim
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x_router_input, x_expert_input):
        batch_size = x_router_input.size(0)
        gating_weights, expert_indices = self.router(x_router_input)
        final_output = torch.zeros(batch_size, self.expert_output_dim, device=x_router_input.device)
        flat_expert_indices = expert_indices.flatten()
        flat_gating_weights = gating_weights.flatten()
        flat_inputs = x_expert_input.repeat_interleave(self.top_k, dim=0)
        expert_outputs = torch.zeros(batch_size * self.top_k, self.expert_output_dim, device=x_router_input.device)
        for i in range(self.num_experts):
            mask = (flat_expert_indices == i)
            if mask.any():
                expert_outputs[mask] = self.experts[i](flat_inputs[mask])
        weighted_outputs = expert_outputs * flat_gating_weights.unsqueeze(1)
        final_output.index_add_(0, torch.arange(batch_size, device=x_router_input.device).repeat_interleave(self.top_k), weighted_outputs)
        return final_output, expert_indices

# --- 5. Main Model: PatentCitationMoEModule ---
class PatentCitationMoEModule(nn.Module):
    def __init__(self, ipc_vocab_size, ipc_embedding_dim, num_roles, role_embedding_dim,
                 bibliometric_feature_dim, num_citation_classes,
                 moe_expert_output_dim, num_experts, moe_top_k, expert_hidden_dims,
                 attention_dim=None):  # 새로운 파라미터 추가
        super().__init__()
        
        # attention_dim 기본값 설정
        if attention_dim is None:
            attention_dim = ipc_embedding_dim
        
        self.ipc_embedding_layer = nn.Embedding(ipc_vocab_size, ipc_embedding_dim, padding_idx=0)
        self.role_embedding_layer = nn.Embedding(num_roles, role_embedding_dim, padding_idx=0)
        
        # 수정된 AttentionPooling 사용 (input_dim과 attention_dim 명시)
        self.attention_pooling_layer = AttentionPooling(
            input_dim=ipc_embedding_dim,  # ipc + role combined 차원
            attention_dim=attention_dim   # output 차원
        )

        # MoE layer에서 router_input_dim을 attention_dim으로 수정
        self.moe_layer = MoELayer(
            router_input_dim=attention_dim,           # 변경: ipc_embedding_dim → attention_dim
            expert_input_dim=bibliometric_feature_dim,
            expert_output_dim=moe_expert_output_dim,
            num_experts=num_experts,
            top_k=moe_top_k,
            expert_hidden_dims=expert_hidden_dims
        )

        self.prediction_head = nn.Linear(moe_expert_output_dim, num_citation_classes)
    
    def initialize_ipc_embeddings(self, word2vec_model, ipc_index_map_func, pad_token='[PAD]', unk_token='[UNK]'):
        embedding_weight = self.ipc_embedding_layer.weight.data
        embedding_dim = embedding_weight.shape[1]

        for ipc_token, idx in ipc_index_map_func.items():
            if idx == 0:  # padding idx는 건너뜀
                continue
            if ipc_token in word2vec_model:
                embedding_weight[idx] = torch.tensor(word2vec_model[ipc_token], dtype=torch.float32)
            elif unk_token in word2vec_model:
                embedding_weight[idx] = torch.tensor(word2vec_model[unk_token], dtype=torch.float32)
            else:
                embedding_weight[idx] = torch.empty(embedding_dim).uniform_(-0.1, 0.1)

        # 패딩 벡터는 0으로 고정
        embedding_weight[0] = torch.zeros(embedding_dim)

    def forward(self, ipc_indices, role_indices, bibliometric_features, ipc_padding_mask=None):
        # 1. IPC + Role embedding
        ipc_embeds = self.ipc_embedding_layer(ipc_indices)  # [B, L, D]
        role_embeds = self.role_embedding_layer(role_indices)  # [B, L, D]
        ipc_role_combined = ipc_embeds + role_embeds       # element-wise sum: [B, L, D]
        
        # 2. Attention pooling → Router input 생성
        # 이제 attention_dim 크기의 벡터가 출력됨
        router_input, attn_weights = self.attention_pooling_layer(
            ipc_role_combined, mask=ipc_padding_mask
        )  # [B, ipc_dim]
        self.latest_attention_weights = attn_weights  # for external access
        
        # 3. Expert input은 bibliometric features 그대로 사용
        expert_input = bibliometric_features  # [B, D_biblio]
        
        # 4. MoE: Router input ≠ Expert input
        moe_output, expert_indices = self.moe_layer(
            x_router_input=router_input,   # [B, ipc_dim]
            x_expert_input=expert_input    # [B, D_biblio]
        )
        
        # 5. Classifier
        logits = self.prediction_head(moe_output)
        return logits, expert_indices
