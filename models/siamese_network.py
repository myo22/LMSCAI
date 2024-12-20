import dezero.functions as F
import dezero.layers as L
from dezero import Model

class SiameseNetwork(Model):
    def __init__(self, input_dim=768, hidden_dim=128):  # input_dim을 768로 수정
        super().__init__()
        # 첫 번째 레이어의 입력 차원을 768으로 설정
        self.fc1 = L.Linear(hidden_dim, nobias=False)  # nobias=False로 설정하면 입력 차원과 일치시킬 수 있음
        self.fc2 = L.Linear(hidden_dim // 2)
        self.fc3 = L.Linear(hidden_dim // 4)

    def feature_extraction(self, x):
        h = F.relu(self.fc1(x))  # 첫 번째 레이어에서 768 차원 받음
        h = F.relu(self.fc2(h))  # 두 번째 레이어는 (hidden_dim // 2) 차원
        h = self.fc3(h)  # 세 번째 레이어는 (hidden_dim // 4) 차원
        return h

    def forward(self, x1, x2):
        h1 = self.feature_extraction(x1)
        h2 = self.feature_extraction(x2)
        distance = F.sum((h1 - h2) ** 2, axis=1)  # 두 벡터의 차이의 제곱합
        similarity = F.exp(-distance)  # 유사도 계산
        return similarity