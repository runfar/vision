import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)