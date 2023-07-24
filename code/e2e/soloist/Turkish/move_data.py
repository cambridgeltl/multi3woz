import os
import shutil

from pathlib import Path
import shutil
import os


if __name__ == '__main__':
    dataset_root = "../../../../data/Turkish"


    if not os.path.exists("data/multi-woz"):
        os.makedirs("data/multi-woz")
        shutil.copy(os.path.join(dataset_root, "data.json"), 'data/multi-woz/')
        shutil.copy(os.path.join(dataset_root, "valListFile.json"), 'data/multi-woz/')
        shutil.copy(os.path.join(dataset_root, "testListFile.json"), 'data/multi-woz/')


    files = os.listdir(dataset_root)
    for fname in files:
        if "db" in fname:
            shutil.copy2(os.path.join(dataset_root, fname), "db")
