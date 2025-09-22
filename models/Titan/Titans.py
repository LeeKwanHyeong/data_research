import torch
import torch.nn as nn
import torch.optim as optim

from model_runner.model_configs import TitanConfig
from models.layers.RevIN import RevIN
from models.layers.Titans_Memory import MemoryEncoder, LMM
from models.layers.TrendCorrector import TrendCorrector


class Model(nn.Module):
    def __init__(self, config: TitanConfig):
        super().__init__()

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

    def forward(self, x):

        # Input Normalization
        x = self.revin_layer(x, 'norm') # [batch, seq_len, input_dim]

        # Encoding
        encoded = self.encoder(x) # [batch, seq_len, d_model]

        # 마지막 타임스텝(hidden state)만 사용하여 예측
        pred = self.output_proj(encoded[:, -1, :]) # [batch, output_horizon]

        # 예측값 역정규화 (Denormalization)
        pred = self.revin_layer(pred.unsqueeze(1), 'denorm').squeeze(1) # [batch, output_horizon]

        return pred

class LMMModel(nn.Module):
    def __init__(self, config: TitanConfig):
        super().__init__()

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
            output_horizon = config.horizon
        )

    # --------------------------- Train ---------------------------
    def forward(self, x, mode = 'train'):
        # 입력 정규화 (Normalization)
        x = self.revin_layer(x, 'norm') # [batch, seq_len, input_dim]
        # 인코딩 (Encoding)
        encoded = self.encoder(x) # [batch, seq_len, d_model]
        B, L, D = encoded.size(0)

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
            enhanced = self.lmm(encoded, memory)
            pred = self.output_proj(enhanced[:, -1, :])
            pred = pred + self.trend_corrector(enhanced[:, -1, :])
            pred = self.revin_layer(pred.unsqueeze(1), 'denorm').squeeze(1)
            return pred

        # --------------------------- Inference ---------------------------
        else:
            # 추론 시에도 Trend 보정을 위한 시퀀스가 필요하므로 encoded 전달은 유지
            pred = self.output_proj(encoded[:, -1, :])
            pred = pred + self.trend_corrector(encoded)
            pred = self.revin_layer(pred.unsqueeze(1), 'denorm').squeeze(1)
            return pred


class FeatureModel(nn.Module):
    def __init__(self, config: TitanConfig, feature_dim = 7):
        super().__init__()
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