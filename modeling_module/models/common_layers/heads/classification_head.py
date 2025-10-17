import torch.nn as nn


class ClassificationHead(nn.Module):
    """
        시계열 백본에서 추출된 feature를 기반으로 다중 클래스 분류를 수행하는 출력 헤드.

        특징:
        - 마지막 patch의 feature만을 사용하여 classification 수행
        - 입력 feature는 [bs, nvars, d_model, num_patch] 구조
        - Flatten 후 Dropout 및 Linear layer로 클래스 score 출력

        예: PatchTST, PatchMixer와 같은 시계열 백본의 출력과 연결
    """
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        """
        생성자

        Parameters:
        - n_vars: 입력 시계열 변수 개수 (예: 센서 수, 피처 수 등)
        - d_model: 백본에서 추출한 feature 차원 수
        - n_classes: 분류할 클래스 수
        - head_dropout: Dropout 확률
        """
        super().__init__()
        self.flatten = nn.Flatten(start_dim = 1)    # [B, C, D] -> [B, C * D]
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, n_classes)

    def forward(self, x):
        """
        Forward 연산

        Parameters:
        - x: Tensor of shape [batch_size, n_vars, d_model, num_patch]
             - 백본에서 출력된 multi-variable feature map

        Returns:
        - y: Tensor of shape [batch_size, n_classes]
             - 각 클래스에 대한 logit 값
        """
        # 마지막 patch의 feature만 사용 → 최신 정보 기반 분류
        x = x[:, :, :, -1]  # [B, n_vars, d_model]

        # flatten: [B, n_vars * d_model]
        x = self.flatten(x)

        # regularization
        x = self.dropout(x)

        # classification
        y = self.linear(x)
        return y

