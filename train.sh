bsub -n 8 -R "rusage[mem=8192]" -R "rusage[ngpus_excl_p=4]" -G ls_polle_s python train.py config/SHARP2020/track2.yaml

