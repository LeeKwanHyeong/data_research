import torch
import torch.nn as nn

from modeling_module.models.common_layers.Attention import AttentionLayer, FullAttention
from modeling_module.models.common_layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, \
    DataEmbedding_wo_pos_temp
from modeling_module.models.common_layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer

'''
    [Vanilla Transformer Based Time-Series]
    Vanilla Transformer based timeseries forecasting model
    구조는 Encoder-Decoder Transformer로, Informer, Autoformer 등 최신 시계열 모델의 기본 형태와 유사하지만
    O(L^2) Complexity의 Full Attention을 그대로 사용.
    
    1. 이 클래스는 Encoder-Decoder Transformer based timeseries forecasting model
    2. Full Attention 사용 -> O(L^2) Complexity
    3. embed_type으로 다양한 임베딩 설정 실험 가능
    4. 출력: 미래 pred_len 길이의 예측 값
'''
class Model(nn.Module):
    '''
        Vanilla Transformer with O(L^2) complexity
    '''
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Data Embedding without positional encoding information
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs, configs.d_model, configs.embed, configs.freq, configs.dropout)

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag = False,
                            factor = configs.factor,
                            attention_dropout = configs.dropout,
                            output_attention = self.output_attention
                        ),
                        d_model = configs.d_model,
                        n_heads = configs.n_heads
                    ),
                    d_model = configs.d_model,
                    d_ff = configs.d_ff,
                    dropout = configs.dropout,
                    activation = configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer = torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag = True,
                            factor = configs.factor,
                            attention_dropout = configs.dropout,
                            output_attention = False
                        ),
                        d_model = configs.d_model,
                        n_heads = configs.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            mask_flag = False,
                            factor = configs.factor,
                            attention_dropout = configs.dropout,
                            output_attention = False
                        ),
                        d_model = configs.d_model,
                        n_heads = configs.n_heads
                    ),
                    d_model = configs.d_model,
                    d_ff = configs.d_ff,
                    dropout = configs.dropout,
                    activation = configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer = torch.nn.LayerNorm(configs.d_model),
            projection = nn.Linear(configs.d_model, configs.c_out, bias = True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask = None, dec_self_mask = None, dec_enc_mask = None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, x_mark_dec)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask = dec_self_mask, cross_mask = dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :] # [B, L, D]