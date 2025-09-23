from model_runner.model_configs import PatchTSTConfig
from models.PatchTST.self_supervised.backbone import PatchTSTEncoder
from models.PatchTST.self_supervised.head import *

"""
(a) Backbone (Supervised)
    Input: Univariate Series
    - Instance Norm(RevIN) + Patching
    - Projection + Position Embedding
    - Transformer Encoder
    - Flatten + Linear Head
    Output: Univariate Series
(b) Backbone (Self-supervised)
    Input: Univariate Series
    - Instance Norm(RevIN) + Patching
    - Projection + Position Embedding
    - Transformer Encoder
    - Linear Layer
    Output: Reconstructed Masked Patches
    
(a), (b) Transformer Encoder
    Inputs
    - Input Embedding
    - Multi-Head Attention
    - Add & Norm
    - Feed Forward
    - Add & Norm
"""

class PatchTSTModel(nn.Module):
    '''
    Output dimension:
        [bs x target_dim x nvars] for prediction
        [bs x target_dim] for regression
        [bs x target_dim] for classification
        [bs x num_patch x n_vars x patch_len] for pretrain
    '''

    def __init__(self,config: PatchTSTConfig):
        super().__init__()

        assert config.head_type in ['pretrain', 'prediction', 'regression', 'classification'],\
        'head type should be either pretrain, prediction, or regression'

        self.config = config

        # Backbone
        self.backbone = PatchTSTEncoder(
            self.config.c_in,
            num_patch = self.config.num_patch,
            patch_len = self.config.patch_len,
            n_layers = self.config.n_layers,
            d_model = self.config.d_model,
            n_heads = self.config.n_heads,
            shared_embedding = self.config.shared_embedding,
            d_ff = self.config.d_ff,
            attn_dropout = self.config.attn_dropout,
            dropout = self.config.dropout,
            act = self.config.act,
            res_attention = self.config.res_attention,
            pre_norm = self.config.pre_norm,
            store_attn = self.config.store_attn,
            pe = self.config.pe,
            learn_pe = self.config.learn_pe,
            verbose = self.config.verbose)

        # Head
        self.n_vars = self.config.c_in
        self.head_type = self.config.head_type

        match self.config.head_type:
            case 'pretrain':
                self.head = PretrainHead(
                    self.config.d_model, self.config.patch_len,
                    self.config.head_dropout
                ) # Custom head passed as a partial func with all its kwargs
            case 'prediction':
                self.head = PredictionHead(
                    self.config.individual, self.n_vars,
                    self.config.d_model, self.config.num_patch,
                    self.config.target_dim, self.config.head_dropout
                )
            case 'regression':
                self.head = RegressionHead(
                    self.n_vars, self.config.d_model,
                    self.config.target_dim, self.config.head_dropout,
                    self.config.y_range)
            case 'classification':
                self.head = ClassificationHead(
                    self.n_vars, self.config.d_model,
                    self.config.target_dim, self.config.head_dropout
                )

    def forward(self, z):
        '''
        z: tensor [bs x num_patch x patch_len]
        '''
        z = self.backbone(z)
        z = self.head(z)
        # z: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification
        #    [bs x num_patch x n_vars x patch_len] for pretrain
        return z





