from glob import glob
import os
import numpy as np
import time
import numpy as np
from tqdm import tqdm

folders = glob('../SHARP_data/track2' + '/*/*')

for folder in folders:
    if not os.path.isdir(folder):
        continue
    files_check = glob(folder + '/*color*.npz')
    for f in tqdm(files_check):
        try:
            f_check = np.load(f)
            if len(f_check.files) < 1:
              #print(f_check.files)
              print(f)
              time.sleep(2)
            f_files = f_check.files
        except:
            print(f)
            time.sleep(2)