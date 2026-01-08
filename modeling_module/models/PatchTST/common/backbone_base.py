# src/modeling_module/models/PatchTST/common/backbone_base.py

import torch
from torch import nn
from modeling_module.models.PatchTST.common.patching import compute_patch_num
from modeling_module.models.PatchTST.common.pos_encoding import positional_encoding, PositionalEncoding


class PatchBackboneBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.c_in = cfg.c_in
        self.d_model = cfg.d_model
        self.patch_len = cfg.patch_len
        self.stride = cfg.stride

        # 1) Patch 개수 계산
        self.patch_num = compute_patch_num(cfg.lookback, cfg.patch_len, cfg.stride, cfg.padding_patch)

        # -----------------------------------------------------------
        # [Refactoring] Input Projection Layer (Mixer)
        # Target Patch + Past Cont Patch + Past Cat Patch -> d_model
        # -----------------------------------------------------------

        # A. Target Dimension
        # 하나의 패치 안에 들어가는 데이터 양: patch_len * c_in
        self.target_input_dim = self.patch_len * self.c_in

        # B. Past Continuous Exo Dimension
        self.d_past_cont = getattr(cfg, 'd_past_cont', 0)
        self.cont_input_dim = self.patch_len * self.d_past_cont

        # C. Past Categorical Exo Dimension
        self.d_past_cat = getattr(cfg, 'd_past_cat', 0)
        self.cat_cardinalities = getattr(cfg, 'cat_cardinalities', [])
        self.d_cat_emb = getattr(cfg, 'd_cat_emb', 8)

        # 범주형 임베딩 레이어 생성
        if self.d_past_cat > 0:
            assert len(self.cat_cardinalities) == self.d_past_cat, \
                f"cat_cardinalities length({len(self.cat_cardinalities)}) must match d_past_cat({self.d_past_cat})"
            self.cat_embs = nn.ModuleList([
                nn.Embedding(c_num, self.d_cat_emb) for c_num in self.cat_cardinalities
            ])
            # 범주형은 임베딩 후 패치화 되므로: patch_len * (변수개수 * 임베딩차원)
            self.cat_input_dim = self.patch_len * (self.d_past_cat * self.d_cat_emb)
        else:
            self.cat_embs = nn.ModuleList()
            self.cat_input_dim = 0

        # D. Total Input Dimension per Patch
        self.total_input_dim = self.target_input_dim + self.cont_input_dim + self.cat_input_dim

        # E. Projection Layer (All -> d_model)
        # 기존 Conv1d 대신 Linear Projection 사용 (유연성을 위해)
        # PatchTST 원본은 unfold + Linear와 동일한 효과를 내기 위해 Conv1d(stride)를 썼지만,
        # 다중 소스 결합을 위해 Unfold -> Concat -> Linear 방식을 채택합니다.
        self.input_proj = nn.Linear(self.total_input_dim, self.d_model)

        # Positional Encoding
        self.pos_enc = positional_encoding(
            pe=getattr(cfg, "pe", "sincos"),
            learn_pe=getattr(cfg, "learn_pe", True),
            q_len=self.patch_num,
            d_model=self.d_model,
        )

    def _process_categorical(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        x_cat: [B, L, n_cat]
        return: [B, L, n_cat * d_emb]
        """
        if self.d_past_cat == 0:
            return None

        embeddings = []
        for i, emb_layer in enumerate(self.cat_embs):
            # x_cat[:, :, i] -> [B, L] -> emb -> [B, L, d_emb]
            embeddings.append(emb_layer(x_cat[..., i]))

        # Concat along embedding dimension -> [B, L, n_cat * d_emb]
        return torch.cat(embeddings, dim=-1)

    def _patchify_and_embed(self, x: torch.Tensor, p_cont: torch.Tensor, p_cat: torch.Tensor) -> torch.Tensor:
        """
        입력:
            x: [B, L, C] (Target)
            p_cont: [B, L, d_past_cont] (Continuous Exo)
            p_cat: [B, L, d_past_cat] (Categorical Exo)
        출력:
            z: [B, Patch_Num, d_model]
        """
        # 1. Prepare Inputs
        # -----------------------------------
        # Unfold를 위해 [B, Channel, Length] 형태로 준비하지 않고,
        # [B, Length, Channel] 상태에서 unfold를 직접 구현하거나 view를 조작합니다.
        # 여기서는 가장 직관적인 Unfold(dim=1) 방식을 사용합니다.

        # 패딩 처리 ('end' padding)
        if self.cfg.padding_patch == 'end':
            # Last patch handling logic needs to be consistent across all inputs
            # 간단하게 ReplicationPad1d 사용 (dim 순서 변경 필요)
            pad_func = nn.ReplicationPad1d((0, self.stride))

            x = pad_func(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L+S, C]
            if p_cont is not None and p_cont.shape[-1] > 0:
                p_cont = pad_func(p_cont.permute(0, 2, 1)).permute(0, 2, 1)
            if p_cat is not None and p_cat.shape[-1] > 0:
                p_cat = pad_func(p_cat.permute(0, 2, 1)).permute(0, 2, 1)

        # 2. Embedding Categorical (Before Patching)
        # -----------------------------------
        # [B, L, n_cat] -> [B, L, n_cat * d_emb]
        p_cat_emb = self._process_categorical(p_cat) if (p_cat is not None and self.d_past_cat > 0) else None

        # 3. Create Patches (Unfold)
        # -----------------------------------
        # unfold: [B, L, C] -> [B, N, C, P] -> flatten -> [B, N, C*P]

        def make_patches(tensor):
            if tensor is None: return None
            # unfold(dimension, size, step)
            # [B, L, C] -> [B, N, C, P]
            patches = tensor.unfold(1, self.patch_len, self.stride)
            B, N, C, P = patches.shape
            return patches.reshape(B, N, C * P)

        x_patches = make_patches(x)  # [B, N, c_in * P]

        features = [x_patches]

        if self.d_past_cont > 0:
            cont_patches = make_patches(p_cont)  # [B, N, d_cont * P]
            features.append(cont_patches)

        if self.d_past_cat > 0 and p_cat_emb is not None:
            cat_patches = make_patches(p_cat_emb)  # [B, N, d_cat_total * P]
            features.append(cat_patches)

        # 4. Concatenate & Project
        # -----------------------------------
        # [B, N, Total_Dim]
        input_vectors = torch.cat(features, dim=-1)

        # [B, N, d_model]
        z = self.input_proj(input_vectors)

        # 5. Add Positional Encoding
        # -----------------------------------
        if self.pos_enc.size(0) < z.size(1):
            new_pe = PositionalEncoding(z.size(1), self.d_model).to(z.device)
            z = z + new_pe.unsqueeze(0)
        else:
            z = z + self.pos_enc[:z.size(1), :].unsqueeze(0)

        return z