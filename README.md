# TH3-GMM-transform
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# Định nghĩa model Transformer
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_proj(src)
        encoded = self.transformer_encoder(src)
        out = self.output_proj(encoded)
        return out, encoded

# ... phần còn lại code của bạn
model = TimeSeriesTransformer().to(device)

# Giúp tránh lỗi phân mảnh bộ nhớ CUDA (nếu dùng GPU)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Định nghĩa mô hình Transformer cho chuỗi thời gian
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=32, nhead=4, num_layers=2, dim_feedforward=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)
    
    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        src = self.input_proj(src)  # (batch_size, seq_len, d_model)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model) vì Transformer yêu cầu thứ tự này
        transformer_out = self.transformer_encoder(src)  # (seq_len, batch_size, d_model)
        transformer_out = transformer_out.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        out = self.output_proj(transformer_out)  # (batch_size, seq_len, 1)
        return out, transformer_out

# Hàm tạo chuỗi dữ liệu cho mô hình học chuỗi thời gian
def create_sequences(data, seq_length=30):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return torch.tensor(np.array(xs)).float(), torch.tensor(np.array(ys)).float()

# Đọc dữ liệu
df = pd.read_csv(r"C:\Users\ADMIN\Downloads\archive (1)\bitcoin_2017_to_2023.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
prices = df[['timestamp', 'close']].dropna().copy()
prices['LogClose'] = np.log(prices['close'])

# Chuẩn hóa dữ liệu LogClose
scaler = StandardScaler()
log_close_scaled = scaler.fit_transform(prices[['LogClose']])

# Phân cụm GMM trên giá trị log(close) đã chuẩn hóa
gmm = GaussianMixture(n_components=4, random_state=42)
prices['GMM_Label'] = gmm.fit_predict(log_close_scaled)

# Vẽ biểu đồ phân cụm GMM
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

# Tạo dữ liệu chuỗi cho Transformer
log_close_np = log_close_scaled.flatten().reshape(-1, 1)  # (N, 1)
X, y = create_sequences(log_close_np, seq_length=30)

# Thêm chiều đặc trưng để phù hợp với đầu vào Transformer (samples, seq_len, input_dim)
X = X.unsqueeze(-1)  # (samples, seq_len, 1)

# Chọn device GPU hoặc CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y = X.to(device), y.to(device)

# Tạo dataloader với batch_size nhỏ để tránh lỗi hết bộ nhớ GPU
batch_size = 16
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Khởi tạo model, optimizer, loss function
model = TimeSeriesTransformer().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Huấn luyện mô hình
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
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # giải phóng bộ nhớ GPU không dùng

# Lấy embedding từ Transformer (trung bình embedding theo chuỗi)
model.eval()
embeddings_list = []
with torch.no_grad():
    for batch_X, _ in dataloader:
        batch_X = batch_X.to(device)
        _, embeddings = model(batch_X)  # embeddings shape: (batch, seq_len, d_model)
        embeddings_mean = embeddings.mean(dim=1).cpu()  # lấy trung bình theo seq_len
        embeddings_list.append(embeddings_mean)

embeddings_all = torch.cat(embeddings_list, dim=0).numpy()

# Phân cụm GMM trên embedding
gmm_embedding = GaussianMixture(n_components=4, random_state=42)
labels = gmm_embedding.fit_predict(embeddings_all)

# Vẽ phân cụm GMM trên không gian embedding
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.scatter(
        embeddings_all[labels == i, 0], embeddings_all[labels == i, 1],
        label=f'Cụm {i}', alpha=0.6
    )
plt.title("Phân cụm GMM trên embedding của Transformer")
plt.xlabel("Embedding dim 1")
plt.ylabel("Embedding dim 2")
plt.legend()
plt.show()
