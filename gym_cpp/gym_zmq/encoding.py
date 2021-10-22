import numpy as np
import torch
import base64
import io


def encode_numpy(array: np.ndarray) -> str:
    b = io.BytesIO()
    torch.save(torch.from_numpy(array), b)
    encoded = base64.b64encode(b.getvalue()).decode('ascii')
    return encoded


def decode_numpy(string: str) -> np.ndarray:
    decoded = base64.b64decode(string)
    buffer = io.BytesIO(decoded)
    return torch.load(buffer).numpy()
