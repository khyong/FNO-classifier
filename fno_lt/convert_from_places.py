import os
import shutil
from pathlib import Path


root = '/home/anonymous/datasets/Places365/data_256'

for dirpath in Path(root).glob('*/*'):
    if '-' in str(dirpath):
        new_dirpath = '/'.join(str(dirpath).split('-'))
        if not os.path.exists(new_dirpath):
            os.makedirs(new_dirpath)
        for filepath in Path(dirpath).glob('*.jpg'):
            shutil.move(str(filepath), new_dirpath)
        os.rmdir(str(dirpath))

