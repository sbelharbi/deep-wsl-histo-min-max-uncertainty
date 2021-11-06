"""
Compress files found in the csv files.
Dataset: camelyon16.
"""

import os
from os.path import join, dirname, abspath
import csv
import fnmatch
import subprocess


import tqdm


def csv_loader(fname):
    """
    Read a *.csv file. Each line contains:
     1. img: str
     2. mask: str or '' or None
     3. label: str

    :param fname: Path to the *.csv file.
    :param rootpath: The root path to the folders of the images.
    :return: List of elements.
    Each element is the path to an image: image path, mask path [optional],
    class name.
    """
    with open(fname, 'r') as f:
        out = [
            [row[0],
             row[1] if row[1] else None,
             row[2]
             ]
            for row in csv.reader(f)
        ]

    return out


def find_files_pattern(fd_in_, pattern_):
    """
    Find paths to files with pattern within a folder recursively.
    :return:
    """
    assert os.path.exists(fd_in_), "Folder {} does not exist " \
                                   ".... [NOT OK]".format(fd_in_)
    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


if __name__ == "__main__":
    files = ['test_s_0_f_0.csv', 'train_s_0_f_0.csv', 'valid_s_0_f_0.csv']
    for f in files:
    	print('checking {}'.format(f))
    	input('HAZAAAA!')
    	root = csv_loader(f)
    	other = csv_loader(join('split_0/fold_0', f))
    	assert len(root) == len(other)
    	i = 0
    	for elroot, elother in zip(root, other):
    		assert elroot == elother
    		print('row {}/[OK]: {}'.format(i, elroot))
    		i += 1
    
