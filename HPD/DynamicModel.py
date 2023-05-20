import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random

# 设置随机种子
torch.manual_seed(42)


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.squeeze(0)
        out = self.fc(h_n)
        return out


def main():
    # 读取原始数据
    data = np.loadtxt('data_output.txt')
    labels = np.loadtxt('label.txt', dtype=str)
    model_path = "lstm_model.pt"

    # 将数据和标签组合成数据块
    data_blocks = np.split(data, len(data) // 5)
    # labels = labels[:len(data_blocks)]

    # 打包数据块和标签
    data_combined = list(zip(data_blocks, labels))

    # 打乱数据顺序
    random.shuffle(data_combined)

    # 解包洗牌后的数据
    shuffled_data_blocks, shuffled_labels = zip(*data_combined)

    # 调整数据形状，使其符合LSTM的输入要求
    X = np.concatenate(shuffled_data_blocks).reshape(-1, 5, 63)  # 数据形状：(样本数, 时间步长, 特征数)
    y = shuffled_labels

    # 将标签转换为整数编码
    label_to_index = {label: i for i, label in enumerate(np.unique(y))}
    print('label_to_index:', label_to_index)
    y = np.array([label_to_index[label] for label in y])

    # 转换为PyTorch的张量
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    # 划分训练集和验证集
    train_ratio = 0.8
    train_size = int(X.shape[0] * train_ratio)

    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)



    # 初始化模型
    input_size = 63
    hidden_size = 64
    output_size = len(label_to_index)
    model = LSTMModel(input_size, hidden_size, output_size)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print("Model saved successfully.")

    # 验证模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            accuracy = correct / total
            print(f"Validation Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()