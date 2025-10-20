import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

def load_and_scale_data(path, features, target):
    df = pd.read_csv(path)
    X = df[features].values
    y = df[target].values

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor, scaler_X