import os
import glob
from envs import HOME_DATA_FOLDER


KD_DIR = os.path.join(HOME_DATA_FOLDER, 'outputs/KD/')
all_res = dict()
for sub_dir in ['pkd_3layer', 'pkd_6layer']:
    all_res[sub_dir] = dict()
    f = os.path.join(KD_DIR, 'race-merge', sub_dir)
    all_sub_dir = glob.glob(f + '/*')
    for ff in all_sub_dir:
        file_name = glob.glob(ff + '/*32.txt')
        if len(file_name) == 0:
            # print(ff, 'missing')
            pass
        else:
            file_name = file_name[0]
            acc = open(file_name).readlines()[0].split('=')[1]
            if float(acc) > 0.6:
                print(file_name, os.path.basename(file_name), float(acc))
                all_res[sub_dir][os.path.basename(file_name)] = float(acc)
    print()


