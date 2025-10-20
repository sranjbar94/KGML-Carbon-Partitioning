import torch
from torch.utils.data import DataLoader, TensorDataset
from kgml_model import KGML_SelfSup, self_supervised_loss

def train_model(X, idx_map, epochs=100, batch_size=64, lr=1e-3, weights=None):
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = KGML_SelfSup(nfeat=X.shape[1]).to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if weights is None:
        weights = {'WUE':1, 'Med':1, 'ENG':1, 'NEE_abs':1, 'NEE_sign':1, 'NIGHT':1, 'Pfvs':1, 'Rfvs':1}

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            X_batch = batch[0].to('cuda' if torch.cuda.is_available() else 'cpu')
            optimizer.zero_grad()
            out = model(X_batch, idx_map)
            loss, stats = self_supervised_loss(out, X_batch, idx_map, weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")
    return model
