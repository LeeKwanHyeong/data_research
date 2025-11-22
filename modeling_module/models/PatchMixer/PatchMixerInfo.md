# PatchMixer: A Patch-Mixing Architecture for Long-Term Time Series Forecasting

> 기반 논문: **PatchMixer: A Patch-Mixing Architecture for Long-Term Time Series Forecasting (Gong et al., HKUST, 2024)**  
> 본 문서는 PatchMixer 논문 내용을 요약하고,  
> 현재 사용자가 구축한 **PatchMixer 기반 수요예측 모델 구조(BaseModel / QuantileModel)** 와의 연계까지 설명합니다.

---

## 1. 연구 배경 및 문제의식

**Long-Term Time Series Forecasting (LTSF)** 은 긴 시계열 구간의 미래값을 예측하는 문제로,  
전통적으로 Transformer 기반 모델(PatchTST, FEDformer 등)이 높은 성능을 보였으나,  
그 **핵심 성능 요인이 Transformer 구조인지, “패치 단위 표현(패치화)” 때문인지**는 명확하지 않았습니다.

> **연구 질문:**  
> “Patch-based Transformer의 성능은 Attention 구조 때문인가,  
> 아니면 Patch 단위의 데이터 전처리 방식 때문인가?”

이를 검증하기 위해 **PatchMixer**는 Attention이 아닌 **Convolution 기반 패치 처리 구조**로  
PatchTST 수준의 성능을 재현하며, **패치 단위 표현이 LTSF의 핵심임**을 보여줍니다.

---

## 2. 핵심 아이디어: Patch-Mixing Design

PatchMixer는 시간축을 **Patch 단위**로 분할하여 입력 시계열의 **로컬 패턴**을 학습하고,  
Depthwise Separable Convolution(DWConv)을 통해 **패치 간의 상호의존성**을 효율적으로 혼합(mixing)합니다.


### Channel Dependency vs. Patch Dependency
- **Channel Mixing (변수 간 혼합)** : 다변량 변수 간 상관관계 학습  
- **Patch Mixing (시간 패치 간 혼합)** : 단일 변수 내 시간 패턴 학습  
→ 실험적으로 **intra-variable dependency(패치 간)**가 더 강함을 확인 → Patch 중심 설계

### Patch Embedding
각 시계열 $$x^{(i)} \in \mathbb{R}^{L}$$ 을 패치 단위로 분리:

$$
x^{(i)}_p = [x^{(i)}_{1:P}, x^{(i)}_{1+S:P+S}, \dots, x^{(i)}_{L-P+1:L}]
$$

- P: patch length  
- S: stride  
- N = $$ \lfloor \frac{L-P}{S} \rfloor + 1 $$: patch 개수  
- 마지막 구간은 **Replicate Padding**을 통해 정보 손실 방지

---

## 3. PatchMixer 아키텍처 개요
![patch_mixer_model.png](../../../img/patch_mixer_model.png)
### (1) 입력 구조
| 단계 | 텐서 형상 | 설명 |
|------|------------|------|
| 입력 | (B, L, N) | 배치, 시계열 길이, 변수 개수 |
| 정규화 | RevIN / InstanceNorm | 분포 이동 완화 |
| 패치 분할 | (B, N, patch_num, patch_size) | 슬라이딩 윈도우 패치화 |
| 선형 투영 | (B, N, patch_num, d_model) | patch_size → d_model |
| Flatten | (B×N, patch_num, d_model) | 변수별 독립처리 |

---

### (2) PatchMixer Block

PatchMixer의 핵심은 **Depthwise + Pointwise Conv**로 구성된 “패치 혼합 블록”입니다.

```python
x = x + DepthwiseConv1d(x)     # 패치 내부 로컬 패턴 (Residual)
x = PointwiseConv1d(x)         # 패치 간 상호작용
x = GELU + BatchNorm + Dropout
```

•	Depthwise Conv  
•	각 채널 독립적 처리로 연산량 감소 (MobileNet/Xception 스타일)  
•	커널 크기 K=8 기준, 지역적 시계열 패턴 포착  
•	Pointwise Conv (1×1)  
•	패치 간 global 관계 학습 (Channel Mixing 효과)  
•	Residual 연결로 안정적 학습 유지
---

### (3) Dual Forecasting Heads

PatchMixer는 Dual Head 구조를 도입해 선형/비선형 패턴을 동시에 학습합니다.

| Head        | 구조                       | 역할                       |
|-------------|--------------------------|--------------------------|
| Linear Head | 단일 Linear                | 장기적 추세(Trend)            |
| MLP Head    | Linear -> GELU -> Linear | 비선형 요동 (Irregularity)    |
| Output      | 두 Head를 합산               | Fine-to-Coarse 시계열 표현 강화 |


---

### (4) Instance Normalization (RevIN)

입력 시계열을 변수 단위로 정규화 후,
예측값 계산 뒤 원래의 평균/표준편차로 되돌리는 구조입니다.

$$
\text{norm}(x) = \frac{x - \mu}{\sigma}, \quad
\text{denorm}(\hat{y}) = \hat{y} \cdot \sigma + \mu
$$

이로써 분포 이동(domain shift) 에 강한 예측 성능 확보.

---

## 4. 논문 실험 결과

| Model                 | Architecture | Speed     | 성능 (MSE 기준) |
|-----------------------|--------------|-----------|--------------------------|
| Transformer(PatchTST) | Attention 기반 | 느림        | 기준 |
| MLP (DLinear)         | 단순 선형        | 빠름        |↓ |
| CNN (PatchMixer)      | DWConv 기반    | 2 ~ 3배 빠름 | ↑ +11.6~21.2% 향상 |

•	MSE/MAE 기준 SOTA 달성  
•	추론 속도 3배, 학습 속도 2배 향상  
•	Attention이 없어도 패치 표현만으로 Transformer 성능 재현

---

## 5. 현재 구현: PatchMixer 구조

### 사용자 커스터마이징 핵심
| 구성                   | 설명                                                  |
|----------------------|-----------------------------------------------------|
| Backbone             | PatchMixerBackbone (RevIN 포함, patch 기반 feature 추출)  |
| Head (BaseModel)     | Linear MLP로 구성된 Point Forecast Head (H-step 예측)     |
| Head (QuantileModel) | Decomposition Quantile Head - (q10, q50, q90) 예측    |
| Feature Branch       | 추가적인 Static/Exogeneous 피처 결합 (FeatureModel)         |
| Loss 함수              | BaseModel -> MSE/MAE, QuantileModel -> Pinball Loss |
| Exogenous Head       | 미래 외생 변수 (온도, 수요 변화율 등 ) 보정용 Linear Head            |


---

### BaseModel 구조
```python
x → PatchMixerBackbone → (B, a*d_model)
   → Linear(128) + Softplus + Dropout
   → Linear(64) + ReLU → Linear(H)
   → Output: (B, H)
```

•	단일 포인트 예측  
•	MSE 또는 MAE 기반 회귀 Loss 사용  
•	외생변수 입력 시 _apply_exo_shift_linear()로 Horizon 보정 수행

---
### QuantileModel 구조
```python
x → MultiScalePatchMixerBackbone(fused_dim=256)
   → DecompQuantileHead(in_dim=256, horizon=H, quantiles=(0.1,0.5,0.9))
   → Output: (B, 3, H)
```

•	다중 스케일 패치 조합으로 장·단기 패턴 동시에 반영  
•	Trend / Seasonal / Irregular decomposition 기반 Quantile 예측  
•	Pinball Loss를 통해 신뢰구간(80%) 기반 불확실성 추정 가능


### 1) PatchMixer **BaseModel** (Point Forecast)

```mermaid
flowchart LR
    subgraph Inference["PatchMixer BaseModel — Forward( )"]
      X["x ∈ ℝ[B,L,N]"] --> RVN["RevIN(norm)"]
      RVN --> BB["PatchMixerBackbone\n• unfold & project W_P\n• depthwise token mixer (Conv1d D-wise)\n• channel mixer (1x1 Conv)\n• var-pooling (mean over N)\n⇒ z ∈ ℝ[B,D]"]
      BB --> EXP["TemporalExpander\n• sinus pos-enc (season_period, harmonics)\n• optional conv\n⇒ x_bhf ∈ ℝ[B,H,F]"]
      EXP --> LN["LayerNorm(F)"]
      LN --> RESID["Residual MLP Head\nLinear→GELU→Linear\n⇒ resid ∈ ℝ[B,H] (zero-mean)"]
      BB --> BASEB["base_head_b(z) ⇒ b ∈ ℝ[B,1]"]
      BB --> BASEM["base_head_m(z) ⇒ m ∈ ℝ[B,1]"]
      BASEB --> BASE["base = b + m⋅t (t: linspace[-1,1]) ⇒ ℝ[B,H]"]
      BASEM --> BASE
      BB --> AGATE["base_gate(z) ⇒ α ∈ (0,1)ᵝ\n(bias init=-2.5)"]
      LN --> GCONV["Step-Gate over H\n• Conv1d k=3,5,dilated3 → concat\n• 1x1 reduce → g_logit\n• τ term + bias/gain/temp\n• clamp & sigmoid → gate ∈ ℝ[B,H]"]
      GCONV --> MIX
      RESID --> MIX
      BASE --> MIX
      AGATE --> MIX
      MIX["y = α·base + (1-α)·(gate·resid)"] --> EXO{"future_exo ?"}
      EXO -->|Yes & normalized| EXA["apply_exo_shift_linear(exo_head)\nadd in normalized space"]
      EXO -->|Yes & raw-unit| EXB["(denorm 이후 add)"]
      EXO -->|No| PASS["(skip)"]
      EXA --> SCALE
      PASS --> SCALE
      SCALE["scale/bias: y = y·out_scale + out_bias"] --> DW["Depthwise(1D) Residual\n(kernel=3) → add\n(local curvature)"]
      DW --> DENORM["RevIN(denorm)"]
      DENORM --> CLAMP{"inference mode ?"}
      CLAMP -->|Yes| NN["clamp_min(0)"]
      CLAMP -->|No| OUT
      NN --> OUT["ŷ ∈ ℝ[B,H]"]
    end

```

### 2) PatchMixer **QuantileModel** (q10, q50, q90)

```mermaid
flowchart LR
    subgraph Forward["PatchMixer QuantileModel — Forward( )"]
      X["x ∈ ℝ[B,L,N]"] --> RVN["RevIN(norm)"]
      RVN --> MSBB["MultiScalePatchMixerBackbone\n• branches: (patch_len,stride,kernel)\n• per-branch proj→per_branch_dim\n• fusion: concat or gated\n⇒ z ∈ ℝ[B,fused_dim]"]
      MSBB --> EXP["TemporalExpander\n⇒ x_bhf ∈ ℝ[B,H,F]"]
      EXP --> DQH["DecompositionQuantileHead(V2)\n• trend(+)\n• Fourier K (n_harmonics)\n• quantiles=[.1,.5,.9]\n⇒ q ∈ ℝ[B,3,H] (normalized)"]
      DQH --> EXO{"future_exo ?"}
      EXO -->|Yes & normalized| EXA["apply_exo_shift_linear\n(add per-horizon)"]
      EXO -->|Yes & raw-unit| EXB["(denorm 이후 add)"]
      EXO -->|No| PASS["(skip)"]
      EXA --> SLICE["Per-h step denorm via RevIN\n(분위수별/시점별 슬라이스)"]
      PASS --> SLICE
      SLICE --> CLAMP{"inference mode ?"}
      CLAMP -->|Yes| NN["clamp_min(0)"]
      CLAMP -->|No| OUT
      NN --> OUT["q_raw ∈ ℝ[B,3,H]"]
    end

```

---

### 3) **PatchMixerBackbone** (단일 스케일)

```mermaid
flowchart TB
    subgraph Backbone["PatchMixerBackbone"]
      X["x ∈ ℝ[B,L,N]"] --> TP["permute → (B,N,L)"]
      TP --> PAD["ReplicationPad1d((0,stride))"]
      PAD --> UNF["unfold(P=patch_len, step=stride)\n⇒ (B,N,A,P)"]
      UNF --> PROJ["W_P: ℝ[P]→ℝ[d_model]\n⇒ (B,N,A,D)"]
      PROJ --> RESH["reshape → (B*N, A, D)"]
      RESH --> PERM["permute → (B*N, D, A)"]
      PERM -->|× e_layers| MIX["PatchMixerLayer (repeat)\n• token mixer: depthwise Conv1d(D-wise)\n• channel mixer: 1x1 Conv\n• residual add"]
      MIX --> FLAT["flatten → (B*N, D·A)"]
      FLAT --> UNV["view → (B,N,D·A)"]
      UNV --> POOL["mean over N\n⇒ z ∈ ℝ[B,D·A]"]
    end

```

---

### 4) **MultiScalePatchMixerBackbone** (병렬 분기 + 융합)

```mermaid
flowchart LR
    subgraph MS["MultiScalePatchMixerBackbone"]
      X["x ∈ ℝ[B,L,N]"] --> B1["Branch #1\n(pl,st,ks)"]
      X --> B2["Branch #2\n(pl,st,ks)"]
      X --> B3["Branch #3\n(pl,st,ks)"]
      B1 --> P1["Linear → per_branch_dim"]
      B2 --> P2["Linear → per_branch_dim"]
      B3 --> P3["Linear → per_branch_dim"]
      P1 --> FUS
      P2 --> FUS
      P3 --> FUS
      FUS{"fusion = 'concat' or 'gated'?"}
      FUS -->|concat| CAT["concat([Bᵢ]) → Linear\n⇒ z ∈ ℝ[B,fused_dim]"]
      FUS -->|gated| GT["softmax(gates)·stack([Bᵢ]) → Linear\n⇒ z ∈ ℝ[B,fused_dim]"]
    end

```

---

### 5) **TemporalExpander**(개념 흐름)

```mermaid
flowchart LR
    Z["z ∈ ℝ[B,D]"] --> REP["repeat/expand to H"]
    REP --> FEAT["concat:
    • sinus/cos seasonals (period, K)
    • optional conv over H
	    ⇒ x_bhf ∈ ℝ[B,H,F]"]

```

---

### 6) **Dynamic Patching** 변형(선택): MoS & Learnable Offset

```mermaid
flowchart TB
    subgraph MoS["DynamicPatcherMoS (Mixture-of-Strides)"]
      X["x ∈ ℝ[B,L,N]"] --> GATE["gate_net(xᵗ) ⇒ softmax K"]
      X --> PBR1["SimpleUnfoldProjector(P1,S1)\n⇒ (B*N,A1,D)"]
      X --> PBR2["SimpleUnfoldProjector(P2,S2)\n⇒ (B*N,A2,D)"]
      X --> PBR3["SimpleUnfoldProjector(P3,S3)\n⇒ (B*N,A3,D)"]
      PBR1 --> PAD1["pad to A_max"]
      PBR2 --> PAD2["pad to A_max"]
      PBR3 --> PAD3["pad to A_max"]
      PAD1 --> SUM
      PAD2 --> SUM
      PAD3 --> SUM
      GATE --> SUM
      SUM["∑(gateₖ · Zₖ) ⇒ (B*N,A_max,D)"]
    end

    subgraph OFF["DynamicOffsetPatcher (Learnable Offsets)"]
      X2["x ∈ ℝ[B,L,N]"] --> OFFH["off_head ⇒ δ ∈ ℝ[B,A]\n(tanh·max_off)"]
      OFFH --> GRID["anchors + δ → sample grid\n(grid_sample on 1D)"]
      GRID --> WP["W_P ⇒ (B*N,A,D)"]
    end

```