"""
Paths and archives management.
"""
import os
import time


# Directories paths
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(DIRNAME)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
OUT_DIR = os.path.join(ROOT_DIR, 'out')

# Dataset paths
DATA_TRAIN = os.path.join(DATA_DIR, 'train_data.pkl')
DATA_VALID = os.path.join(DATA_DIR, 'val_data.pkl')

# Training paths
DEFAULT_LOSSES_PATH = os.path.join(MODELS_DIR, 'losses.pickle')
DEFAULT_WEIGHTS_PATH = os.path.join(MODELS_DIR, 'weights.pt')
DEFAULT_PARAMETERS_PATH = os.path.join(MODELS_DIR, 'parameters.pickle')

# Testing paths
DEFAULT_PREDICTIONS_DIR = os.path.join(OUT_DIR, 'predictions')


def create_dirs() -> None:
    """Creates directories if needed."""
    for path in (MODELS_DIR, OUT_DIR):
        if not os.path.exists(path):
            os.mkdir(path)


def generate_filename(filename: str) -> str:
    """Generates a filename using time.

    Args:
        filename (str): string template. Example: 'model-{}.pt'.

    Returns:
        str: filename with time.
    """
    timestr = time.strftime('%Y%m%d')
    return filename.format(timestr)


def generate_model_filename() -> str:
    """Generates model filename using time.

    Returns:
        str: model filename with time.
    """
    filename = generate_filename('model-{}.pt')
    return os.path.join(MODELS_DIR, filename)


def generate_log_filename() -> str:
    """Generates log filename using time.

    Returns:
        str: log filename with time.
    """
    filename = generate_filename('log-{}.pickle')
    return os.path.join(MODELS_DIR, filename)
