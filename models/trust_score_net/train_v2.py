import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os


class TrustScoreNetV2(nn.Module):
    """
    V2 Trust Score Neural Network.
    Takes exactly 2 verified inputs:
        1. signed_deberta_score  (-1.0 to +1.0)
        2. numeric_deviation     (-1.0 to 1.0)
    Outputs: Trust Score (0 to 100)
    """
    def __init__(self):
        super(TrustScoreNetV2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Outputs 0.0 to 1.0
        )

    def forward(self, x):
        return self.net(x) * 100.0  # Scale to 0-100


def train_v2():
    data_path = os.path.join(os.path.dirname(__file__), "v2_training_data.csv")

    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Run dataset_mapper.py first!")
        return

    print("1. Loading Genuine Mapped Data...")
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} rows with {df['signed_deberta_score'].nunique()} unique DeBERTa scores.")

    # Only 2 verified features
    X = df[['signed_deberta_score', 'numeric_deviation']].values
    Y = df['TARGET_TRUST_SCORE'].values / 100.0

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

    # Use DataLoader for proper mini-batch training (prevents memorization)
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Training on: {device}")

    model = TrustScoreNetV2().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    epochs = 300
    print("\n2. Training V2 Model (Mini-Batch SGD)...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X) / 100.0
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        scheduler.step()

        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / batches
            print(f"   Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.6f}")

    # Save V2 Checkpoint
    save_path = os.path.join(os.path.dirname(__file__), "checkpoint_v2.pt")
    torch.save(model.state_dict(), save_path)

    # Quick sanity test
    model.eval()
    with torch.no_grad():
        # Test: High support + no price deviation = should be high trust
        test1 = torch.tensor([[0.85, 0.0]], dtype=torch.float32).to(device)
        # Test: High contradiction + massive price deviation = should be low trust
        test2 = torch.tensor([[-0.90, 0.85]], dtype=torch.float32).to(device)

        score1 = model(test1).item()
        score2 = model(test2).item()

    print(f"\n3. SUCCESS! V2 Model saved to {save_path}")
    print(f"\n   SANITY CHECK:")
    print(f"   Genuine product  (deberta=+0.85, dev=0.00) -> Trust: {score1:.1f}/100")
    print(f"   Scam product     (deberta=-0.90, dev=0.85) -> Trust: {score2:.1f}/100")

    if score1 > 70 and score2 < 30:
        print("   [PASS] Model is LEARNING correct patterns (not memorizing)!")
    else:
        print("   [FAIL] WARNING: Model may need more diverse data or tuning.")


if __name__ == "__main__":
    train_v2()
