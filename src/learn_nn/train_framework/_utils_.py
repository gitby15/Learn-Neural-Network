import torch
from torch import nn
from pathlib import Path
import time
FOLDER_PATH = Path(__package__).resolve()

_prefix = time.strftime("%Y%m%d_%H_%M_%S")
LOG_FILE_PATH = FOLDER_PATH / "temp" / f"{_prefix}_log.txt"
MODEL_FILE_PATH = FOLDER_PATH / "temp" / f"{_prefix}_model.pth"
print(f"== LOG START, LOG FILE: {LOG_FILE_PATH} ==")

INFERENCE_START_TAG = '=============INFERENCE_START============='
INFERENCE_END_TAG = '=============INFERENCE_END============='

def _append_file(line: str):
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(line + '\n')

def log_output_line(line: str):
    # print(line)
    _append_file(line)


def save_model(model: nn.Module):
    MODEL_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_FILE_PATH)


def get_device():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return DEVICE

if __name__ == "__main__":
    print("path: package: ", FOLDER_PATH)