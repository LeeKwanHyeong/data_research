import torch
import torch.nn as nn
import math
'''
    [PositionalEmbedding]
    Desc: Positional Embedding(위치 임베딩)는 Transformer 모델에서 사용하는 기법으로, 시퀀스 데이터의 순서를
          모델이 이해할 수 있도록 위치 정보를 추가하는 역할
    1. Transformer에서 위치 정보를 추가하기 위해 사용하는 모듈
    2. 사인/코사인 함수를 사용하여 주기적이고 일반화 가능한 위치 인코딩 생성
    3. 학습하지 않는 고정 텐서로 구현, forward 시 입력 길이에 맞게 슬라이싱
'''
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        # 위치 임베딩을 저장할 텐서 초기화 (max_len x d_model)
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False # 위치 벡터는 학습하지 않음

        # [0, 1, 2, ..., max_len-1] 위치 인덱스 생성 후 차원 추가 (max_len x 1)
        position = torch.arange(0, max_len).float().unsqueeze(1)

        # 주파수 스케일링: 각 차원마다 다른 주기를 갖도록 계산
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # 짝수 차원에는 sin, 홀수 차원에는 cos 값 채우기
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1 x max_len x d_model) 형태로 변환하여 배치 차원 추가
        pe = pe.unsqueeze(0)

        # 학습 파라미터는 아니지만 모델에 buffer로 등록 (state_dict에 저장됨)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 입력 x의 시퀀스 길이에 맞는 positional encoding 반환
        return self.pe[:, :x.size(1)]

'''
    [TokenEmbedding]
    Desc: TokenEmbedding은 입력 시계열(또는 토큰 시퀀스)을 임베딩 차원(d_model)으로 변환하기 위해 1D Convolution을 사용하는 레이어
    1. TokenEmbedding은 Conv1d를 이용해 입력 특성(c_in)을 d_model 차원으로 매핑
    2. Circular Padding + kernel_size=3으로 인접 관계와 주기성 반영
    3. 출력 형태는 (batch, seq_len, d_model) → Transformer 입력에 적합
'''
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2

        # Conv1d: 입력 채널 (c_in) -> 출력 채널 (d_model)로 매핑
        self.tokenConv = nn.Conv1d(
            in_channels = c_in,
            out_channels = d_model,
            kernel_size = 3,
            padding = padding,
            padding_mode = 'circular', # 시계열 주기성 반영
            bias = False)

        # Kaiming Normal 초기화 (Leaky ReLU 기반)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_in', nonlinearity = 'leaky_relu')

    def forward(self, x):
        # x: (batch, seq_len, c_in) -> (batch, c_in, seq_len)로 변환 후 Conv 적용
        x = self.tokenConv(x.permute(0, 2, 1)).tranpose(1, 2)

        # Result: (batch, seq_len, d_model)
        return x

'''
    [FixedEmbedding]
    Desc: FixedEmbedding은 입력 토큰 (또는 feature index)에 대해 학습되지 않는 고정 임베딩을 제공하는 클래스
          Transformer에서 일반적으로 사용하는 학습 가능한 임베딩과 달리, 이 방식은 Positional Encoding과 동일한 
          Sin/Cos Pattern을 이용하여 비학습적(non-learnable) 임베딩 생성
    1. FixedEmbedding은 입력 인덱스를 사인/코사인 기반 고정 임베딩으로 매핑
    2. 학습 불가능(requires_grad=False), 주기적 패턴을 이용해 일반화 가능
    3. 출력 형태: (batch, seq_len, d_model)
'''
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        # 고정 임베딩 초기화 (c_in x d_model)
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False # 학습 불가능

        # 위치 인덱스 (c_in x 1)
        position = torch.arange(0, c_in).float().unsqueeze(1)
        # Frequency Scaling Calculation
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model)).exp()

        # 짝수 차원: sin, 홀수 차원: cos 적용
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        # nn.Embedding Layer 생성 후, 계산된 w로 초기화
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad = False)

    def forward(self, x):
        # 입력 인덱스를 고정 임베딩으로 변환 (역전파 없음)
        return self.emb(x).detach()

'''
    [TemporalEmbedding]
    Desc: TemporalEmbedding은 시계열 데이터의 시간 관련 요소(월, 일, 요일, 시, 분)를 임베딩 벡터로 변환하는 모듈
          Transformer 기반 시계열 모델에서는 단순히 값(time index)만 입력하는 것이 아니라, 시간적 패턴을 인코딩하기 위해 활용
    
    1. TemporalEmbedding은 시간 요소(월, 일, 요일, 시, 분)를 임베딩 벡터로 변환
    2. embed_type: 'fixed'(사인/코사인 기반) 또는 'learnable'(Conv 기반) 선택 가능
    3. 출력: (batch, seq_len, d_model), 모든 시간 임베딩을 합산하여 Transformer 입력으로 사용
'''
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type = 'fixed', freq = 'h'):
        super(TemporalEmbedding, self).__init__()

        # 시간 요소별 값의 범위 설정
        minute_size = 4     # 분 (0~3, freq = 't'에서 사용)
        hour_size = 24      # 시 (0 ~ 23)
        weekday_size = 7    # 요일 (0 ~ 6)
        day_size = 32       # 일 (1~31, 0은 패딩용)
        month_size = 13     # 월 (1~12, 0은 패딩용)

        # 임베딩 타입 선택 (고정 vs 학습)
        Embed = FixedEmbedding if embed_type == 'fixed' else TokenEmbedding
        # 분 단위 주기 사용 시 minute_embed 추가
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        # 나머지 시간 요소별 임베딩
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        # 입력 x: (batch, seq_len, 5) [month, day, weekday, hour, minute]
        x = x.long()

        # 각 시간 요소별 임베딩 계산
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        # 모든 시간 임베딩을 합산하여 반환
        return hour_x + weekday_x + day_x + month_x + minute_x

'''
    [TimeFeatureEmbedding]
    Desc: TimeFeatureEmbedding은 시계열 데이터에서 시간 관련 특성을 연속형 값으로 입력받아 선형 변환을 통해
          모델 차원(d_model)으로 매핑하는 모듈.
          
    1. TimeFeatureEmbedding은 연속형 시간 특성을 d_model 차원으로 선형 매핑
    2. freq_map을 통해 입력 feature 개수를 동적으로 결정
    3. 학습 가능한 임베딩 (Linear Layer), 출력 shape: (batch, seq_len, d_model)
'''
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type = 'timeF', freq = 'h'):
        super(TimeFeatureEmbedding, self).__init__()

        # freq_map: 주어진 frequency 별 입력 feature 개수
        freq_map = {
            'h': 4, # hour: [month, day, weekday, hour]
            't': 5, # minute: + minute
            's': 6, # second: + second
            'm': 1, # month 단위
            'a': 1, # annual 단위
            'w': 2, # week 단위
            'd': 3, # day 단위
            'b': 3  # business day
        }
        # Input Feature Size
        d_inp = freq_map[freq]

        # Linear Transform Layer: (d_inp -> d_model), bias False
        self.embed = nn.Linear(d_inp, d_model, bias = False)

    def forward(self, x):
        # Input: (batch, seq_len, d_inp)
        # Output: (batch, seq_len, d_model)
        return self.embed(x)

'''
    [DataEmbedding]
    Desc: Transformer기반 시계열 모델에서 최종 입력 임베딩을 생성하는 통합 모듈.
          시계열 데이터는 값(Value) + 위치(Position) + 시간 특성(time features) 세 가지 정보를 함께 사용해야 한다.
          이 클래스는 이를 합쳐서 모델 입력으로 사용할 수 있도록 처리.
    Composition:
          1. Value Embedding (TokenEmbedding): 원본 시계열 값 (c_in)을 Conv1d 기반으로 d_model 차원으로 변환
          2. Positional Embedding (PositionalEmbedding): Transformer가 순서를 알 수 있도록 위치 정보 추가
          3. Temporal/Time Feature Embedding
            - TemporalEmbedding: discrete한 시간 요소 (month, day 등)를 임베딩
            - TimeFeatureEmbedding: 연속형 시간 특성을 선형 변환
          4. Dropout: 정규화 목적으로 최종 임베딩에 Dropout 적용
    Final Output:
        Embedding = ValueEmbedding(x) + TemporalEmbedding(X_mark) + PositionalEmbedding(x)
        
    1. DataEmbedding은 Value + Position + Temporal 정보를 결합한 Transformer 입력 벡터 생성
    2. TemporalEmbedding vs TimeFeatureEmbedding은 embed_type으로 결정
    3. 출력 shape: (batch, seq_len, d_model), Dropout 적용
'''
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type = 'fixed', freq = 'h', dropout = 0.1):
        super(DataEmbedding, self).__init__()

        # 값(value) Embedding: Conv1d
        self.value_embedding = TokenEmbedding(c_in = c_in, d_model = d_model)
        # 위치(position) Embedding: Sin/Cos
        self.position_embedding = PositionalEmbedding(d_model = d_model)
        # 시간(time feature) Embedding: 선택적 (Temporal or TimeFeature)
        self.temporal_embedding = TemporalEmbedding(
            d_model = d_model,
            embed_type = embed_type,
            freq = freq
        ) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model = d_model,
            embed_type = embed_type,
            freq = freq
        )
        # Dropout: 정규화 목적
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, x_mark):
        # x: (batch, seq_len, c_in)
        # x_mark: (batch, seq_len, feature_size)
        # 최종 임베딩: Value + Time + Position
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

'''
    [DataEmbedding_wo_pos]
    Desc: DataEmbedding_wo_pos는 DataEmbedding과 유사하지만, Positional Embedding을 제외한 임베딩 모듈.
          Transformer 모델에서 위치 정보(Positional Encoding)는 중요한 역할을 하지만, 특정 실험이나 모델 구조에서는 
          위치 정보를 제거하고 Value + Temporal 만 사용하는 경우가 있음.
          
'''
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type = 'fixed', freq = 'h', dropout = 0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        # 값(value) Embedding: Conv1d
        self.value_embedding = TokenEmbedding(c_in = c_in, d_model = d_model)
        # 위치(position) Embedding: Sin/Cos
        self.position_embedding = PositionalEmbedding(d_model = d_model)
        # 시간(time feature) Embedding: 선택적 (Temporal or TimeFeature)
        self.temporal_embedding = TemporalEmbedding(
            d_model = d_model,
            embed_type = embed_type,
            freq = freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model = d_model,
            embed_type = embed_type,
            freq = freq
        )
        # Dropout: 정규화 목적
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, x_mark):
        # x: (batch, seq_len, c_in)
        # x_mark: (batch, seq_len, feature_size)
        # 최종 임베딩: Value + Time
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_wo_pos_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type = 'fixed', freq = 'h', dropout = 0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in = c_in, d_model = d_model)
        self.position_embedding = PositionalEmbedding(d_model = d_model)
        self.temporal_embedding = TimeFeatureEmbedding(
            d_model = d_model,
            embed_type = embed_type,
            freq = freq
        ) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model = d_model,
            embed_type = embed_type,
            freq = freq
        )
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type = 'fixed', freq = 'h', dropout = 0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in = c_in, d_model = d_model)
        self.position_embedding = PositionalEmbedding(d_model = d_model)
        self.temporal_embedding = TemporalEmbedding(
            d_model = d_model,
            embed_type = embed_type,
            freq = freq
        ) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model = d_model,
            embed_type = embed_type,
            freq = freq
        )
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)