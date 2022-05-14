from glob import glob
import os
import tqdm
import multiprocessing as mp
from multiprocessing import Pool



print('Finding Paths to convert (from .npz to .obj files).')
paths = glob('../track2_testdata/*/*/*.npz')


print('Start converting.')
def convert(path):
    outpath = path[:-4] + '.obj'

    cmd = 'python -m sharp convert {} {}'.format(path,outpath)
    os.system(cmd)
if __name__=='__main__':

    p = Pool(mp.cpu_count())
    for _ in tqdm.tqdm(p.imap_unordered(convert, paths), total=len(paths)):
        pass
