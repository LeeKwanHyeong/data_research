import torch
import torch.nn as nn
import torch.optim as optim

from models.layers.RevIN import RevIN
from models.layers.Titans_Memory import MemoryEncoder, LMM
from models.layers.TrendCorrector import TrendCorrector


class TitanConfig:
    batch_size = 128              # 한 번의 학습/추론에서 처리할 시계열 샘플 개수 (배치 크기)
    lookback = 12                 # 입력 시계열의 과거 타임스텝 길이 (모델 입력 시 고려하는 시점 수)
    input_dim = 1                  # 각 타임스텝에서의 피처(feature) 개수 (예: 단변량=1, 다변량>1)
    d_model = 64                   # 모델 내부 임베딩 차원 (토큰/패치가 변환되는 은닉 공간의 차원)
    n_layers = 2                   # Transformer 또는 Titan 블록의 층(layer) 개수
    n_heads = 4                    # Multi-Head Attention에서 병렬로 사용하는 어텐션 헤드 수
    d_ff = 256                     # 피드포워드 네트워크(FFN) 내부 차원 (확장된 은닉층 크기)
    contextual_mem_size = 64       # MAC(Memory-as-Context)에서 사용하는 컨텍스트 메모리 슬롯 개수
    persistent_mem_size = 16       # MAC에서 사용하는 영구 메모리(persistent memory) 슬롯 개수
    horizon = 60            # 예측할 미래 타임스텝 길이 (출력 시점 수)


class Model(nn.Module):
    def __init__(self, config = TitanConfig):
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
        x = self.revin_layer(x) # [batch, seq_len, input_dim]

        # Encoding
        encoded = self.encoder(x) # [batch, seq_len, d_model]

        # 마지막 타임스텝(hidden state)만 사용하여 예측
        pred = self.output_proj(encoded[:, -1, :]) # [batch, output_horizon]

        # 예측값 역정규화 (Denormalization)
        pred = self.revin_layer(pred.unsqueeze(1), 'denorm').squeeze(1) # [batch, output_horizon]

        return pred

class LMMModel(nn.Module):
    def __init__(self, config = TitanConfig):
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

    def forward(self, x, mode = 'train'):
        # 입력 정규화 (Normalization)
        x = self.revin_layer(x, 'norm') # [batch, seq_len, input_dim]
        # 인코딩 (Encoding)
        encoded = self.encoder(x) # [batch, seq_len, d_model]
        B = encoded.size(0)

        if mode == 'train':
            # 학습 시, Encoder Layer의 Contextual Memory 수집
            contextual_memory = []
            for layer in self.encoder.layers:
                mem = layer.attn.contextual_memory
                if mem is not None:
                    contextual_memory.append(mem)

            if len(contextual_memory) > 0:
                mem = contextual_memory[-1] # [M, d_model]
                if mem.dim() == 2:
                    mem = mem.unsqueeze(0).expand(B, -1, -1) # [B, M, d_model]로 broadcast
                elif mem.dim() == 3 and mem.size(0) != B:
                    mem = mem[:1].expand(B, -1, -1)
                memory = mem
            else:
                # Context Memory가 없으면, 안전한 기본값 사용
                # Ex) 입력과 동일 차원 대체 or 0 memory 1 slot
                memory = encoded.new_zeros(B, 1, encoded.size(-1))


            # # 가장 마지막 레이어의 Contextual Memory 수집
            # memory = contextual_memory[-1] if contextual_memory else torch.empty_like(encoded)

            # LMM(Local Memory Matching) 적용
            enhanced = self.lmm(encoded, memory) # [batch, seq_len, d_model]

            # 미래 시점 예측
            pred = self.output_proj(enhanced[:, -1, :]) # [batch, output_horizon]

            # 추세 보정
            pred = pred + self.trend_corrector(enhanced)

            # 역 정규화
            pred = self.revin_layer(pred.unsqueeze(1), 'denorm').squeeze(1) # [batch, output_horizon]
            return pred

        else:
            # 추론 시, LMM 생략 -> 단순 Encoder 출력 사용
            pred = self.output_proj(encoded[:, -1, :]) # [batch, output_horizon]
            pred = pred + self.trend_corrector(encoded) # 추세 보정
            pred = self.revin_layer(pred.unsqueeze(1), 'denorm').unsqueeze(1) # 역정규화
            return pred

class FeatureModel(nn.Module):
    def __init__(self, config = TitanConfig, feature_dim = 7):
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