from glob import glob
import os
import numpy as np
import time
import numpy as np

folders = glob('../SHARP_data/track2' + '/*/*')

for folder in folders:
	if not os.path.isdir(folder):
		continue
	files = glob(folder + '/*0.8.npz')
	count = 0
	for f in files:
   f_check = np.load(f)
   if len(f_check.files) == 0:
     print(f)
		#count += 1
	#if folder.__contains__('gt'):
		#if count != 1:
			##print(folder, count)
			#time.sleep(1)
	#else:
		#if count != 16:
		#	print(folder, count)
			#time.sleep(1)
