class PatchMixerConfigMonthly:
    enc_in = 1
    lookback = 36
    batch_size = 128
    horizon = 48
    patch_len = 16
    stride = 8
    mixer_kernel_size = 8
    d_model = 16
    head_dropout = 0.1
    e_layers = 2

class PatchMixerConfigWeekly:
    enc_in = 1
    lookback = 54
    batch_size = 128
    horizon = 27
    patch_len = 48
    stride = 8
    mixer_kernel_size = 8
    d_model = 16
    head_dropout = 0.1
    e_layers = 2

class TitanConfigMonthly:
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

class TitanConfigWeekly:
    batch_size = 128              # 한 번의 학습/추론에서 처리할 시계열 샘플 개수 (배치 크기)
    lookback = 54                 # 입력 시계열의 과거 타임스텝 길이 (모델 입력 시 고려하는 시점 수)
    input_dim = 1                  # 각 타임스텝에서의 피처(feature) 개수 (예: 단변량=1, 다변량>1)
    d_model = 64                   # 모델 내부 임베딩 차원 (토큰/패치가 변환되는 은닉 공간의 차원)
    n_layers = 2                   # Transformer 또는 Titan 블록의 층(layer) 개수
    n_heads = 4                    # Multi-Head Attention에서 병렬로 사용하는 어텐션 헤드 수
    d_ff = 256                     # 피드포워드 네트워크(FFN) 내부 차원 (확장된 은닉층 크기)
    contextual_mem_size = 64       # MAC(Memory-as-Context)에서 사용하는 컨텍스트 메모리 슬롯 개수
    persistent_mem_size = 16       # MAC에서 사용하는 영구 메모리(persistent memory) 슬롯 개수
    horizon = 27            # 예측할 미래 타임스텝 길이 (출력 시점 수)