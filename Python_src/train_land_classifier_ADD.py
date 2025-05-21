# -*- coding: utf-8 -*-
import os
import json
import configparser
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ===== 1. INIファイル読み込み =====
config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")
patch_dir = config["OUTPUT"]["PATCH_DIR"]

# ===== 2. Dataset定義 =====
class LandPatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        self.label_map = {}

        for idx, land_type in enumerate(sorted(os.listdir(root_dir))):
            land_path = os.path.join(root_dir, land_type)
            if os.path.isdir(land_path):
                self.label_map[land_type] = idx
                for fname in os.listdir(land_path):
                    if fname.endswith(".tif"):
                        self.samples.append(os.path.join(land_path, fname))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# ===== 3. データ変換 + DataLoader =====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = LandPatchDataset(patch_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ===== 4. CNNモデル定義 =====
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ===== 5. モデル初期化 or 継続学習対応 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "land_classifier_model.pth"
model = SimpleCNN(num_classes=len(dataset.label_map)).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("✅ 既存モデルを読み込みました（ファインチューニングモード）。")
else:
    print("🆕 新規モデルとして学習を開始します。")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===== 6. 学習ループ（追加学習） =====
num_epochs = 5  # 継続学習用に少なめ
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

# ===== 7. モデル保存 =====
torch.save(model.state_dict(), model_path)
print("💾 モデルを保存しました。")

# ===== 8. ラベルマップ保存（マージ方式） =====
label_map_path = "label_map.json"
if os.path.exists(label_map_path):
    with open(label_map_path, "r", encoding="utf-8") as f:
        old_map = json.load(f)
    # マージ（重複を避ける）
    old_map.update(dataset.label_map)
    merged_map = old_map
else:
    merged_map = dataset.label_map

with open(label_map_path, "w", encoding="utf-8") as f:
    json.dump(merged_map, f, ensure_ascii=False, indent=2)
print("🗂 ラベルマップを保存しました。")
