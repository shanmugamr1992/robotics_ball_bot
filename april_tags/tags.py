import numpy as np

 # Bot x -> April tag -z
 # Bot y -> April tag x
 # Bot z -> April tag -y
class Tags:
    tags = {
      1: np.array([
        [0, 0, -1, 0], # Bot x
        [1, 0, 0, 12], # Bot y
        [0, -1, 0, 12], # Bot z
        [0, 0, 0, 1],
      ], np.float32),
      2: np.array([
        [0, 0, -1, 0],
        [1, 0, 0, 36],
        [0, -1, 0, 12],
        [0, 0, 0, 1],
      ], np.float32),
      3: np.array([
        [0, 0, -1, 0],
        [1, 0, 0, 60],
        [0, -1, 0, 12],
        [0, 0, 0, 1],
      ], np.float32),

      # Bot x -> April tag x
      # Bot y -> April tag z
      # Bot z -> April tag -y
      4: np.array([
        [1, 0, 0, 12], # Bot x
        [0, 0, 1, 72], # Bot y
        [0, -1, 0, 12], # Bot z
        [0, 0, 0, 1],
      ], np.float32),
      5: np.array([
        [1, 0, 0, 36], # Bot x
        [0, 0, 1, 72], # Bot y
        [0, -1, 0, 12], # Bot z
        [0, 0, 0, 1],
      ], np.float32),
      6: np.array([
        [1, 0, 0, 60], # Bot x
        [0, 0, 1, 72], # Bot y
        [0, -1, 0, 12], # Bot z
        [0, 0, 0, 1],
      ], np.float32)
    }
