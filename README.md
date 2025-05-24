# các thưu viện cần thiết
pip install pandas matplotlib scikit-learn
!pip install hmmlearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from hmmlearn import hmm
# đọc dữ liệu và xử lý dl
df = pd.read_csv("bitcoin_2017_to_2023.csv")
print(df.columns)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
prices = df[['timestamp', 'close']].dropna().copy()
prices['LogClose'] = np.log(prices['close'])
# Chuẩn hóa và phân cụm GMM trên log giá
scaler = StandardScaler()
log_close_scaled = scaler.fit_transform(prices[['LogClose']])
gmm = GaussianMixture(n_components=4, random_state=42)
prices['GMM_Label'] = gmm.fit_predict(log_close_scaled)
# biêur đò

plt.figure(figsize=(12,6))
for i in range(4):
    plt.plot(prices['timestamp'][prices['GMM_Label']==i],
             prices['close'][prices['GMM_Label']==i],
             label=f'Cụm {i}')
plt.legend()
plt.title("GMM phân cụm theo log(close)")
plt.xlabel("Ngày")
plt.ylabel("Giá Bitcoin")
plt.show()
# Chuẩn bị dữ liệu chuỗi thời gian cho mô hình Transformer
def create_sequences(data, seq_length=30):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return torch.tensor(np.array(xs)).float(), torch.tensor(np.array(ys)).float()
# tạo dự leieuj transform
# Flatten sẽ thành 1 chiều (N,), ta cần reshape lại thành (N, 1) để chuẩn bị cho Transformer
log_close_np = log_close_scaled.flatten().reshape(-1, 1)
X, y = create_sequences(log_close_np, seq_length=30)
X = X.unsqueeze(-1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y = X.to(device), y.to(device)
log_close_np = log_close_scaled.flatten()
X, y = create_sequences(log_close_np, seq_length=30)
X = X.unsqueeze(-1)  # (samples, seq_len, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = X.to(device)
# mô hình transform
class TimeSeriesTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(1,16 )
        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.decoder = nn.Linear(16, 1)
    def forward(self, src):
        src = self.input_proj(src)
        encoded = self.encoder(src)
        out = self.decoder(encoded)
        return out, encoded
from torch.utils.data import DataLoader, TensorDataset

# Giả sử X, y đã là tensor với shape đúng
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # batch_size nhỏ để tiết kiệm RAM

model = TimeSeriesTransformer().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        y_pred, _ = model(batch_X)
        loss = loss_fn(y_pred[:, -1, 0], batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
model = TimeSeriesTransformer().to(device)
...
# Train loop
...
with torch.no_grad():
    _, embeddings = model(X)
embeddings_mean = embeddings.mean(dim=1).cpu().numpy()
# Trích xuất embedding từ Transformer
with torch.no_grad():
    _, embeddings = model(X)
embeddings_mean = embeddings.mean(dim=1).cpu().numpy()
# Dùng GMM phân cụm trên embedding
gmm = GaussianMixture(n_components=4, random_state=42)
labels = gmm.fit_predict(embeddings_mean)
# Vẽ biểu đồ trên embedding
# Giảm chiều xuống 2D
pca = PCA(n_components=2)
embedding_2d = pca.fit_transform(embeddings_all)
# Phân cụm GMM trên embedding đã giảm chiều
gmm_embedding = GaussianMixture(n_components=4, random_state=42)
labels = gmm_embedding.fit_predict(embedding_2d)
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.scatter(
        embedding_2d[labels == i, 0], embedding_2d[labels == i, 1],
        label=f'Cụm {i}', alpha=0.6
    )
plt.title("GMM trên embedding Transformer (giảm chiều bằng PCA)")
plt.xlabel("Thành phần chính 1")
plt.ylabel("Thành phần chính 2")
plt.legend()
plt.show()



