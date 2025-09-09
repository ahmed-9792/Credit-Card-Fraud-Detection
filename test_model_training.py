from src.model_training import train_and_save_models
import numpy as np

def test_train_models():
    X = np.random.rand(50, 10)
    y = np.random.randint(0, 2, 50)
    models = train_and_save_models(X, y, save_dir="models")
    assert len(models) > 0
