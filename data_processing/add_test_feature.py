import numpy as np
import tqdm
from glob import glob
from tqdm import tqdm
import os
split = np.load("../SHARP_data/track2/split.npz")


ROOT2 = '../SHARP_data/track3/val_partial'
test_files = glob(ROOT2 + '/*/*.obj')
print(test_files)
test_paths = []
print(f"The number of validation: {len(test_files)}")
for f in tqdm(test_files, desc = "validation"):
    test_paths.append(os.path.splitext(f)[0])
    test_texture = np.array(test_paths, dtype = np.str_)

np.savez("../SHARP_data/track2/split.npz", train = split['train'], val = split['val'], test = split['test'], test_texture = test_texture)

