# import torch
# print('torch version:' + torch.__version__)              # 설치된 torch 버전 확인
# print('cuda version:' + torch.version.cuda)             # '12.2' 출력되면 정상
# print('cudnn version:' + str(torch.backends.cudnn.version())) # 8.x 또는 9.x 출력되면 정상
# print('cuda available:' + str(torch.cuda.is_available()))      # True 나오면 GPU 사용 가능


import tensorflow as tf
print("\nTensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
print("GPU devices:", tf.config.list_physical_devices('GPU'))