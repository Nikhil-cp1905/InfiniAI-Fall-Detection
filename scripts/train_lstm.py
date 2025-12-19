import torch
import torch.nn as nn
import numpy as np
import os

class FallLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def train():
    X = torch.tensor(np.load("outputs/X.npy"), dtype=torch.float32)
    y = torch.tensor(np.load("outputs/y.npy"), dtype=torch.long)

    model = FallLSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        out = model(X)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} Loss {loss.item():.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/fall_lstm.pt")
    print("✅ LSTM model saved to models/fall_lstm.pt")


if __name__ == "__main__":
    train()

