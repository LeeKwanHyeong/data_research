import torch
import torch.nn as nn
import torch.optim as optim

from modeling_module.models.Titan.backbone import MemoryEncoder, PatchMemoryEncoder
from modeling_module.models.Titan.common.configs import TitanConfig
from modeling_module.models.Titan.common.decoder import TitanDecoder
from modeling_module.models.Titan.common.memory import LMM
from modeling_module.models.common_layers.RevIN import RevIN
from modeling_module.models.common_layers.TrendCorrector import TrendCorrector

def _ensure_config_instance(config):
    return config() if isinstance(config, type) else config

class Model(nn.Module):
    def __init__(self, config: TitanConfig):
        super().__init__()
        self.model_name = 'Titan BaseModel'

        # RevIN (Reversible Instance Normalization) 레이어
        # - 입력 시계열 데이터를 인스턴스 단위로 정규화하여 분포 변화를 완화
        # - affine=True: 정규화 후 학습 가능한 scale/shift 적용
        # - subtract_last=True: 시계열 마지막 값을 기준으로 변동성을 보정
        self.revin_layer = RevIN(
            num_features = config.input_dim,
            affine = True,
            subtract_last = True
        )

        # MemoryEncoder (Titan의 핵심 인코더)
        # - 시계열 패턴을 Transformer 기반 구조로 인코딩
        # - MAC(Memory-as-Context) 구조 적용 → contextual/persistent 메모리 활용
        self.encoder = MemoryEncoder(
            config.input_dim,
            config.d_model,
            config.n_layers,
            config.n_heads,
            config.d_ff,
            config.contextual_mem_size,
            config.persistent_mem_size
        )

        # 출력 투영 레이어
        # - 인코더의 은닉 상태(d_model)를 미래 예측 시점(output_horizon) 길이로 변환
        self.output_proj = nn.Linear(config.d_model, config.horizon)

        # exo가 있을 경우: (H, exo_dim) -> (H)로 만ㅁ드는 작은 헤드
        self.exo_dim = getattr(config, 'exo_dim', 0)
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1)    # time-distributed: 마지막에 squeeze(-1)
            )
        else:
            self.exo_head = None

    def forward(self, x, future_exo: torch.Tensor | None = None):

        # Input Normalization
        x = self.revin_layer(x, 'norm') # [batch, seq_len, input_dim]

        # Encoding
        encoded = self.encoder(x) # [batch, seq_len, d_model]

        if encoded.dim() != 3:
            raise RuntimeError(f"Encoder must return (B,L,D). Got {tuple(encoded.shape)}")


        # 마지막 타임스텝(hidden state)만 사용하여 예측
        pred = self.output_proj(encoded[:, -1, :])  # [batch, output_horizon]

        if (self.exo_head is not None) and (future_exo is not None):
            # future_exo : [B, H, exo_dim]
            exo_term = self.exo_head(future_exo).squeeze(-1)    # [B, H]
            pred = pred + exo_term


        # 예측값 역정규화 (Denormalization)
        pred = self.revin_layer(pred.unsqueeze(1), 'denorm').squeeze(1) # [batch, output_horizon]

        return pred

class LMMModel(nn.Module):
    def __init__(self, config: TitanConfig):
        super().__init__()
        self.model_name = 'Titan LMMModel'

        # RevIN (Reversible Instance Normalization)
        # - 입력 시계열을 인스턴스 단위로 정규화하여 분포 변화 완화
        # - affine=True → 정규화 후 학습 가능한 스케일·시프트 적용
        # - subtract_last=True → 마지막 값 기준 변동성 보정
        self.revin_layer = RevIN(
            num_features = config.input_dim,
            affine = True,
            subtract_last= True
        )

        # MemoryEncoder (Titan의 메인 인코더)
        # - Transformer 기반 시계열 인코더
        # - MAC(Memory-as-Context) 구조를 활용해 컨텍스트/영구 메모리를 Attention에 결합
        self.encoder = MemoryEncoder(
            config.input_dim,
            config.d_model,
            config.n_layers,
            config.n_heads,
            config.d_ff,
            config.contextual_mem_size,
            config.persistent_mem_size
        )

        # LMM (Local Memory Matching) 모듈
        # - 인코더 은닉 상태와 메모리 간 로컬 패턴 매칭 수행
        # - top_k=5 → 메모리에서 상위 5개의 관련 패턴만 선택
        self.lmm = LMM(
            d_model = config.d_model,
            top_k = 5
        )

        # 출력 투영 레이어
        # - d_model 차원을 output_horizon 길이로 변환
        # - Softplus 활성화 → 예측값이 항상 0 이상이 되도록 보장
        self.output_proj = nn.Sequential(
            nn.Linear(config.d_model, config.horizon),
            nn.Softplus()
        )

        # Trend Corrector
        # - LMM 결과 또는 인코딩 결과에 대해 추세(trend) 보정 수행
        # - 모델이 장기적인 패턴을 안정적으로 예측하도록 보조
        self.trend_corrector = TrendCorrector(
            d_model = config.d_model,
            out_dim = config.horizon
        )

        self.exo_dim = getattr(config, 'exo_dim', 0)
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1)
            )
        else:
            self.exo_head = None

    # --------------------------- Train ---------------------------
    def forward(self, x, future_exo: torch.Tensor, mode = 'train'):
        # 입력 정규화 (Normalization)
        x = self.revin_layer(x, 'norm') # [batch, seq_len, input_dim]
        # 인코딩 (Encoding)
        encoded = self.encoder(x) # [batch, seq_len, d_model]
        if encoded.dim() != 3:
            raise RuntimeError(f"Encoder must return (B,L,D). Got {tuple(encoded.shape)}")
        B, L, D = encoded.shape

        # (선택) 안전 가드: 만약 인코더가 [B, D, L]을 주는 구현이라면 전치
        if encoded.dim() != 3:
            raise RuntimeError(f"Encoder output must be 3D [B,L,D], got shape={tuple(encoded.shape)}")

        # 예: [B, D, L]로 오는 케이스 방어
        if L < D and hasattr(self, "d_model") and D != getattr(self, "d_model"):
            # 인코더가 [B, D, L]을 반환한 것으로 추정 → 전치
            encoded = encoded.transpose(1, 2)  # [B, L, D]
            B, L, D = encoded.shape

        if mode == 'train':
            # ------ Memory 조립 강화: contextual + persistent + encoded ------
            mem_chunks = []

            # 1) contextual memory
            ctx = None
            for layer in self.encoder.layers:
                m = getattr(layer.attn, 'contextual_memory', None)
                if m is not None and m.numel() > 0:
                    ctx = m

            if ctx is not None:
                if ctx.dim() == 2:                          # [M, D] -> [B, M, D]
                    ctx = ctx.unsqueeze(0).expand(B, -1, -1)
                elif ctx.dim() == 3 and ctx.size(0) != B:   # [B', M, D] -> [B, M, D]
                    ctx = ctx.expand(B, -1, -1)
                mem_chunks.append(ctx)

            # 2) persistent memory
            pm = getattr(self.encoder.layers[-1].attn, 'persistent_memory', None) #[M_p, D]
            if pm is not None and pm.numel() > 0:
                pmB = pm.unsqueeze(0).expand(B, -1, -1)
                mem_chunks.append(pmB)

            # 3) fallback: encoded token 자체도 메모리에 포함
            mem_chunks.append(encoded)                  # [B, L, D]
            memory = torch.cat(mem_chunks, dim = 1)     # [B, M_tot, D]


            # 4) Final Step
            enhanced = self.lmm(encoded, memory)  # (B, L, D)
            pred = self.output_proj(enhanced[:, -1, :])  # (B, out_dim)
            pred = pred + self.trend_corrector(enhanced[:, -1, :])  # 2D 입력 허용

        # --------------------------- Inference ---------------------------
        else:
            # 추론 시에도 Trend 보정을 위한 시퀀스가 필요하므로 encoded 전달은 유지
            pred = self.output_proj(encoded[:, -1, :])
            pred = pred + self.trend_corrector(encoded)

        if (self.exo_head is not None) and (future_exo is not None):
            exo_term = self.exo_head(future_exo).squeeze(-1)
            pred = pred + exo_term

        y = self.revin_layer(pred.unsqueeze(1), 'denorm').squeeze(1)
        return y

"""
x --(RevIN norm)--> PatchMemoryEncoder --> enc:[B, L', D]
                               │
                               ├─ LMM(enc, memory) → enhanced:[B, L', D]
                               └─ TrendCorrector(enc) → Δtrend:[B, H]
        y_core = Linear(D→H)(enhanced[:, -1, :])
        y_hat  = RevIN denorm(y_core + Δtrend)
"""
class PatchLMMModel(nn.Module):
    """
    Patch 기반 Titan + LMM 보강 모델
    입력:
        x: [B, L, C]
        future_exo: Optional[[B, H, exo_dim]]
    출력:
        y_hat: [B, H] (H = config.horizon or output_horizon)
    """
    def __init__(self, config):
        super().__init__()
        config = _ensure_config_instance(config)
        self.config = config

        self.horizon = getattr(config, 'horizon', getattr(config, 'output_horizon', None))
        assert self.horizon is not None, 'config.horizon (or output_horizon)가 필요합니다.'

        d_model = config.d_model

        # RevIN: 입력 분포 표준화/역정규화
        self.revin_layer = RevIN(
            num_features=config.input_dim,
            affine=True,
            subtract_last=True
        )

        # Patch 기반 인코더 (PatchEmbed1D + Mixer + MemoryAttention Backbone)
        self.encoder = PatchMemoryEncoder(
            input_dim=config.input_dim,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            contextual_mem_size=config.contextual_mem_size,
            persistent_mem_size=config.persistent_mem_size,
            # ---- Patch/Mixer 관련 (config에 없으면 기본값)
            patch_len=getattr(config, 'patch_len', 12),
            patch_stride=getattr(config, 'patch_stride', 8),
            n_mixer_blocks=getattr(config, 'n_mixer_blocks', 2),
            mixer_hidden=getattr(config, 'mixer_hidden', 2 * config.d_model),
            mixer_kernel=getattr(config, 'mixer_kernel', 7),
            dropout=getattr(config, 'dropout', 0.1),
        )

        # LMM: Local Pattern Matching 보강
        self.lmm = LMM(
            d_model=d_model,
            top_k=getattr(config, 'lmm_top_k', 5)
        )

        # 출력 헤드: 마지막 토큰으로 H-step 생성 (DMS)
        nonneg = getattr(config, 'nonneg_head', True)
        head = [nn.Linear(d_model, self.horizon)]
        if nonneg:
            head.append(nn.Softplus())
        self.output_proj = nn.Sequential(*head)

        # Trend/Seasonality 보정
        self.trend_corrector = TrendCorrector(
            d_model=d_model,
            out_dim=self.horizon
        )

        # LMM Memory Source 선택: 'encoded' (안정) | 'context'
        self.lmm_memory_source = getattr(config, 'lmm_memory_source', 'encoded')

        # ===== Exogenous 입력(optional) 지원 추가 =====
        self.exo_dim = getattr(config, 'exo_dim', 0)
        if self.exo_dim > 0:
            # time-distributed linear: [B,H,exo_dim] -> [B,H,1] -> squeeze(-1)
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1)
            )
        else:
            self.exo_head = None

    def _get_lmm_memory(self, enc: torch.Tensor) -> torch.Tensor:
        """
        LMM에 공급할 memory Tensor.
        - 기본값: enc  (형상 일치/안정)
        - 옵션: 인코더 레이어의 contextual_memory 사용('context')
        return: [B, M, D]
        """
        B, Lp, D = enc.shape
        if self.lmm_memory_source == 'context':
            ctx = None
            for layer in self.encoder.layers:
                mem = getattr(layer.attn, 'contextual_memory', None)
                if mem is not None:
                    ctx = mem   # [M, D]
            if ctx is not None and ctx.numel() > 0:
                return ctx.unsqueeze(0).expand(B, -1, -1)   # [B, M, D]
            # fallback to encoded
        return enc  # [B, L', D]

    def forward(self, x, future_exo: torch.Tensor | None = None, mode: str = 'train'):
        """
        x: [B, L, C]
        future_exo: Optional[[B, H, exo_dim]]
        return: [B, H]
        """
        # 입력 정규화
        x = self.revin_layer(x, 'norm')  # [B, L, C]

        # Patch+Memory Encoding
        enc = self.encoder(x)            # [B, L', D]
        if enc.dim() != 3:
            raise RuntimeError(f"Encoder must return (B,L,D). Got {tuple(enc.shape)}")

        # LMM 보강
        memory = self._get_lmm_memory(enc)  # [B, M, D]
        if memory.dim() == 2:  # [M, D] -> [B, M, D]
            memory = memory.unsqueeze(0).expand(enc.size(0), -1, -1)
        enhanced = self.lmm(enc, memory)    # [B, L', D]

        # Head (마지막 토큰으로 H-step)
        y_core = self.output_proj(enhanced[:, -1, :])   # [B, H]

        # Trend/Seasonality 보정
        y = y_core + self.trend_corrector(enc)          # [B, H]

        # ===== Exogenous term (optional) =====
        if (self.exo_head is not None) and (future_exo is not None):
            # future_exo: [B, H, exo_dim]
            if future_exo.dim() != 3 or future_exo.size(1) != self.horizon:
                raise ValueError(f"future_exo must be [B, H, exo_dim] with H={self.horizon}, got {tuple(future_exo.shape)}")
            exo_term = self.exo_head(future_exo).squeeze(-1)  # [B, H]
            y = y + exo_term

        # 역정규화
        y = self.revin_layer(y.unsqueeze(1), 'denorm').squeeze(1)   # [B, H]
        return y

# ---- Seq2Seq 모델 (Encoder + Causal Decoder) ----
class LMMSeq2SeqModel(nn.Module):
    """
    MemoryEncoder + light Causal Decoder로 H-step 한 번에 출력(DMS).
    """
    def __init__(self, config: TitanConfig):
        super().__init__()
        self.model_name = 'Titan LMMSeq2SeqModel'


        config = _ensure_config_instance(config)
        self.config = config
        self.horizon = getattr(config, 'horizon', getattr(config, 'output_horizon', None))
        assert self.horizon is not None, 'config.horizon is required'

        self.revin_layer = RevIN(num_features = config.input_dim, affine = True, subtract_last = True)

        self.encoder = MemoryEncoder(
            config.input_dim,
            config.d_model,
            config.n_layers,
            config.n_heads,
            config.d_ff,
            config.contextual_mem_size,
            config.persistent_mem_size
        )

        n_dec_layers = getattr(config, 'n_dec_layers', 1)
        dec_dropout = getattr(config, 'dec_dropout', 0.1)
        exo_dim = getattr(config, 'exo_dim', 0)

        self.decoder = TitanDecoder(
            d_model = config.d_model,
            n_layers = n_dec_layers,
            n_heads = config.n_heads,
            d_ff = config.d_ff,
            dropout = dec_dropout,
            horizon = self.horizon,
            exo_dim = exo_dim,
            causal = True
        )

        self.output_proj = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Softplus() if getattr(config, 'nonneg_head', True) else nn.Identity()
        )

        self.trend_corrector = TrendCorrector(
            d_model = config.d_model,
            out_dim = self.horizon
        )

        self.exo_dim = getattr(config, 'exo_dim', 0)
        if self.exo_dim > 0:
            self.exo_head = nn.Sequential(
                nn.Linear(self.exo_dim, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1)
            )
        else:
            self.exo_head = None

    def forward(self, x, future_exo: torch.Tensor | None = None, mode: str = 'train'):
        """
        x:          [B, L, C]
        future_exo: [B, L, exo_dim] or None
        return:     [B, H]
        """
        # 1) RevIN Normalization
        x = self.revin_layer(x, 'norm')

        # 2) Encode
        encoded = self.encoder(x)                   # [B, L, D]

        if encoded.dim() != 3:
            raise RuntimeError(f"Encoder must return (B,L,D). Got {tuple(encoded.shape)}")


        # 3) Decode
        decoded = self.decoder(encoded, future_exo)     # [B, H, D]

        # 4) Head
        pred = self.output_proj(decoded).squeeze(-1)   # [B, H]

        # 5) Trend Corrector
        pred = pred + self.trend_corrector(encoded)       # [B, H]

        if (self.exo_head is not None) and (future_exo is not None):
            exo_term = self.exo_head(future_exo).squeeze(-1)
            pred = pred + exo_term

        # 6) RevIN Denormalization
        pred = self.revin_layer(pred.unsqueeze(1), 'denorm').squeeze(1)   # [B, H]
        return pred

class FeatureModel(nn.Module):
    def __init__(self, config: TitanConfig, feature_dim = 7):
        super().__init__()
        self.model_name = 'Titan FeatureModel'

        # MemoryEncoder (Titan 기반 인코더)
        # - 시계열 데이터를 Transformer 기반 구조로 인코딩
        # - MAC(Memory-as-Context) 구조로 컨텍스트/영구 메모리 활용
        self.encoder = MemoryEncoder(
            config.input_dim,
            config.d_model,
            config.n_layers,
            config.n_heads,
            config.d_ff,
            config.contextual_mem_size,
            config.persistent_mem_size
        )

        # 부가 피처 투영 레이어
        # - feature_x(예: 요일, 월, 이벤트 플래그 등) 벡터를 d_model 차원으로 매핑
        self.feature_proj = nn.Linear(feature_dim, config.d_model)

        # 출력 투영 레이어
        # - 최종 결합된 은닉 상태를 미래 시계열 길이(output_horizon)로 변환
        self.output_proj = nn.Linear(config.d_model, config.horizon)

    def forward(self, x, feature_x):
        # 시계열 인코딩
        # - 입력 시계열 x → [batch, seq_len, d_model]
        encoded = self.encoder(x)

        if encoded.dim() != 3:
            raise RuntimeError(f"Encoder must return (B,L,D). Got {tuple(encoded.shape)}")


        # 부가 피처 인코딩
        # - feature_x → [batch, feature_dim]
        # - Linear projection → [batch, d_model]
        # - unsqueeze(1) → [batch, 1, d_model] (시계열 시점 차원 추가)
        feature_embed = self.feature_proj(feature_x).unsqueeze(1)

        # 시계열 마지막 시점 은닉 상태와 부가 피처 임베딩 결합
        # - encoded[:, -1, :] → [batch, d_model] (마지막 타임스텝의 인코딩 벡터)
        # - feature_embed → [batch, 1, d_model]
        combined = encoded[:, -1, :] + feature_embed # [batch, 1, d_model]

        # 결합된 벡터를 미래 시계열(output_horizon)로 변환
        return self.output_proj(combined.squeeze(1)) # [batch, output_horizon]


class TestTimeMemoryManager:
    def __init__(self, model, lr = 1e-4):
        # 관리할 대상 모델 (Titan, LMMModel 등)
        self.model = model

        # 최적화 설정 (Adam Optimizer)
        # - 테스트 시점 적응(Test-Time Adaptation) 단계에서 파라미터 업데이트 수행
        self.optimizer = optim.Adam(model.parameters(), lr = lr)

        # 손실 함수 (Mean Squared Error)
        # - 예측값과 실제값의 차이를 제곱 평균으로 계산
        self.loss_fn = nn.MSELoss()

    def add_context(self, new_context):
        # 모델 파라미터가 있는 장치(GPU/CPU)로 new_context 이동
        device = next(self.model.parameters()).device

        # 그래디언트 추적 방지 (detach) → 메모리 업데이트 시 역전파 불필요
        new_context = new_context.to(device).detach()  # detach 추가

        # 모델 인코더의 모든 레이어에 컨텍스트 메모리 업데이트
        # - block.attn.update_contextual_memory()는 MAC 구조에서
        #   Contextual Memory 슬롯에 새로운 정보 저장
        for block in self.model.encoder.layers:
            block.attn.update_contextual_memory(new_context)

    def adapt(self, x_new, y_new, steps=1):

        # 입력과 타겟 데이터를 모델 파라미터와 동일한 장치로 이동
        device = next(self.model.parameters()).device
        x_new = x_new.to(device).float()
        y_new = y_new.to(device).float()

        # 모델을 학습 모드로 전환 (dropout, batch norm 등 학습 시 동작)
        self.model.train()
        for _ in range(steps):
            pred = self.model(x_new)  # 새 forward
            loss = self.loss_fn(pred, y_new)
            self.optimizer.zero_grad()
            loss.backward()  # retain_graph 필요 없음 (새 forward)
            self.optimizer.step()
        return loss.item()