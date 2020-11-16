"""
Splits the following dataset into k-folds:
1. GlaS.
2. Caltech-UCSD-Birds-200-2011
"""

import glob
from os.path import join, relpath, basename, splitext, isfile
import os
import traceback
import random
import sys
import math
import csv
import copy
import getpass
import fnmatch

import yaml
import numpy as np
from scipy.io import loadmat
from PIL import Image, ImageChops
import tqdm


from tools import chunk_it, Dict2Obj, announce_msg

import reproducibility


def split_valid_glas(args):
    """
    Create a validation/train sets in GlaS dataset.
    csv file format: relative path to the image, relative path to the mask, class (str: benign, malignant).

    :param args:
    :return:
    """
    classes = ["benign", "malignant"]
    all_samples = []
    # Read the file Grade.csv
    baseurl = args.baseurl
    with open(join(baseurl, "Grade.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # get rid of the header.
        for row in reader:
            # not sure why they thought it is a good idea to put a space before the class. Now, I have to get rid of
            # it and possibly other hidden spaces ...
            row = [r.replace(" ", "") for r in row]
            assert row[2] in classes, "The class `{}` is not within the predefined classes `{}`".format(row[2], classes)
            all_samples.append([row[0], row[2]])

    assert len(all_samples) == 165, "The number of samples {} do not match what they said (165) .... [NOT " \
                                    "OK]".format(len(all_samples))

    # Take test samples aside. They are fix.
    test_samples = [s for s in all_samples if s[0].startswith("test")]
    assert len(test_samples) == 80, "The number of test samples {} is not 80 as they said .... [NOT OK]".format(len(
        test_samples))

    all_train_samples = [s for s in all_samples if s[0].startswith("train")]
    assert len(all_train_samples) == 85, "The number of train samples {} is not 85 as they said .... [NOT OK]".format(
        len(all_train_samples))

    benign = [s for s in all_train_samples if s[1] == "benign"]
    malignant = [s for s in all_train_samples if s[1] == "malignant"]

    # Split
    splits = []
    for i in range(args.nbr_splits):
        for _ in range(1000):
            random.shuffle(benign)
            random.shuffle(malignant)
        splits.append({"benign": copy.deepcopy(benign),
                       "malignant": copy.deepcopy(malignant)}
                      )

    # Create the folds.
    def create_folds_of_one_class(lsamps, s_tr, s_vl):
        """
        Create k folds from a list of samples of the same class, each fold contains a train, and valid set with a
        predefined size.

        Note: Samples are expected to be shuffled beforehand.

        :param lsamps: list of paths to samples of the same class.
        :param s_tr: int, number of samples in the train set.
        :param s_vl: int, number of samples in the valid set.
        :return: list_folds: list of k tuples (tr_set, vl_set, ts_set): where each element is the list (str paths)
                 of the samples of each set: train, valid, and test, respectively.
        """
        assert len(lsamps) == s_tr + s_vl, "Something wrong with the provided sizes."

        # chunk the data into chunks of size ts (the size of the test set), so we can rotate the test set.
        list_chunks = list(chunk_it(lsamps, s_vl))
        list_folds = []

        for i in range(len(list_chunks)):
            vl_set = list_chunks[i]

            right, left = [], []
            if i < len(list_chunks) - 1:
                right = list_chunks[i + 1:]
            if i > 0:
                left = list_chunks[:i]

            leftoverchunks = right + left

            leftoversamples = []
            for e in leftoverchunks:
                leftoversamples += e

            tr_set = leftoversamples
            list_folds.append((tr_set, vl_set))

        return list_folds

    # Save the folds.
    # Save the folds into *.csv files.
    def dump_fold_into_csv(lsamples, outpath):
        """
        Write a list of RELATIVE paths into a csv file.
        Relative paths allow running the code an any device.
        The absolute path within the device will be determined at the running time.

        csv file format: relative path to the image, relative path to the mask, class (str: benign, malignant).

        :param lsamples: list of str of relative paths.
        :param outpath: str, output file name.
        :return:
        """
        with open(outpath, 'w') as fcsv:
            filewriter = csv.writer(fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for name, clas in lsamples:
                filewriter.writerow([name + ".bmp", name + "_anno.bmp", clas])

    def create_one_split(split_i, test_samples, benign, malignant, nbr_folds):
        """
        Create one split of k-folds.

        :param split_i: int, the id of the split.
        :param test_samples: list, list of test samples.
        :param benign: list, list of train benign samples.
        :param malignant: list, list of train maligant samples.
        :param nbr_folds: int, number of folds [the k value in k-folds].
        :return:
        """
        vl_size_benign = math.ceil(len(benign) * args.folding["vl"] / 100.)
        vl_size_malignant = math.ceil(len(malignant) * args.folding["vl"] / 100.)

        list_folds_benign = create_folds_of_one_class(benign, len(benign) - vl_size_benign, vl_size_benign)
        list_folds_malignant = create_folds_of_one_class(malignant, len(malignant) - vl_size_malignant,
                                                         vl_size_malignant)

        assert len(list_folds_benign) == len(list_folds_malignant), "We didn't obtain the same number of fold" \
                                                                    " .... [NOT OK]"
        assert len(list_folds_benign) == 5, "We did not get exactly 5 folds, but `{}` .... [ NOT OK]".format(
            len(list_folds_benign))
        print("We found {} folds .... [OK]".format(len(list_folds_malignant)))

        outd = args.fold_folder
        for i in range(nbr_folds):
            out_fold = join(outd, "split_" + str(split_i) + "/fold_" + str(i))
            if not os.path.exists(out_fold):
                os.makedirs(out_fold)

            # dump the test set
            dump_fold_into_csv(test_samples, join(out_fold, "test_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the train set
            train = list_folds_malignant[i][0] + list_folds_benign[i][0]
            # shuffle
            for t in range(1000):
                random.shuffle(train)

            dump_fold_into_csv(train, join(out_fold, "train_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the valid set
            valid = list_folds_malignant[i][1] + list_folds_benign[i][1]
            dump_fold_into_csv(valid, join(out_fold, "valid_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the seed
            with open(join(out_fold, "seed.txt"), 'w') as fx:
                fx.write("MYSEED: " + os.environ["MYSEED"])

        with open(join(outd, "readme.md"), 'w') as fx:
            fx.write("csv format:\nrelative path to the image, relative path to the mask, class "
                     "(str: benign, malignant).")

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write("csv format:\nrelative path to the image, relative path to the mask, class "
                 "(str: benign, malignant).")
    # Creates the splits
    for i in range(args.nbr_splits):
        create_one_split(i, test_samples, splits[i]["benign"], splits[i]["malignant"], args.nbr_folds)

    print("All GlaS splitting (`{}`) ended with success .... [OK]".format(args.nbr_splits))


def split_valid_Caltech_UCSD_Birds_200_2011(args):
    """
    Create a validation/train sets in Caltech_UCSD_Birds_200_2011 dataset.
    csv file format: relative path to the image, relative path to the mask, class (str).

    :param args:
    :return:
    """
    baseurl = args.baseurl
    classes_names, classes_id = [], []
    # Load the classes: id class
    with open(join(baseurl, "CUB_200_2011", "classes.txt"), "r") as fcl:
        content = fcl.readlines()
        for el in content:
            el = el.rstrip("\n\r")
            idcl, cl = el.split(" ")
            classes_id.append(idcl)
            classes_names.append(cl)
    # Load the images and their id.
    images_path, images_id = [], []
    with open(join(baseurl, "CUB_200_2011", "images.txt"), "r") as fim:
        content = fim.readlines()
        for el in content:
            el = el.strip("\n\r")
            idim, imgpath = el.split(" ")
            images_id.append(idim)
            images_path.append(imgpath)

    # Load the image labels.
    images_label = (np.zeros(len(images_path)) - 1).tolist()
    with open(join(baseurl, "CUB_200_2011", "image_class_labels.txt"), "r") as flb:
        content = flb.readlines()
        for el in content:
            el = el.strip("\n\r")
            idim, clid = el.split(" ")
            # find the image index correspd. to the image id
            images_label[images_id.index(idim)] = classes_names[classes_id.index(clid)]

    # All what we need is in images_label, images_path. classes_names will be used later to convert class name into
    # integers.
    assert len(images_id) == 11788, "We expect Caltech_UCSD_Birds_200_2011 dataset to have 11788 samples. We found {}" \
                                    ".... [NOT OK]".format(len(images_id))
    all_samples = list(zip(images_path, images_label))  # Not used.

    # Split into train and test.
    all_train_samples = []
    test_samples = []
    with open(join(baseurl, "CUB_200_2011", "train_test_split.txt"), "r") as flb:
        content = flb.readlines()
        for el in content:
            el = el.strip("\n\r")
            idim, st = el.split(" ")
            img_idx = images_id.index(idim)
            img_path = images_path[img_idx]
            img_label = images_label[img_idx]
            filename, file_ext = os.path.splitext(img_path)
            mask_path = join("segmentations", filename + ".png")
            img_path = join("CUB_200_2011", "images", img_path)
            assert os.path.isfile(join(args.baseurl, img_path)), "Image {} does not exist!".format(
                join(args.baseurl, img_path))
            assert os.path.isfile(join(args.baseurl, mask_path)), "Mask {} does not exist!".format(
                join(args.baseurl, mask_path))
            pair = (img_path, mask_path, img_label)
            if st == "1":  # train
                all_train_samples.append(pair)
            elif st == "0":  # test
                test_samples.append(pair)
            else:
                raise ValueError("Expected 0 or 1. Found {} .... [NOT OK]".format(st))

    print("Nbr. ALL train samples: {}".format(len(all_train_samples)))
    print("Nbr. test samples: {}".format(len(test_samples)))

    assert len(all_train_samples) + len(test_samples) == 11788, "Something is wrong. We expected 11788. Found: {}" \
                                                                ".... [NOT OK]".format(
        len(all_train_samples) + len(test_samples))

    # Keep only the required classes:
    if args.nbr_classes is not None:
        fyaml = open(args.path_encoding, 'r')
        contyaml = yaml.load(fyaml)
        keys_l = list(contyaml.keys())
        indexer = np.array(list(range(len(keys_l)))).squeeze()
        select_idx = np.random.choice(indexer, args.nbr_classes, replace=False)
        selected_keys = []
        for idx in select_idx:
            selected_keys.append(keys_l[idx])

        # Drop samples outside the selected classes.
        tmp_all_train = []
        for el in all_train_samples:
            if el[2] in selected_keys:
                tmp_all_train.append(el)
        all_train_samples = tmp_all_train

        tmp_test = []
        for el in test_samples:
            if el[2] in selected_keys:
                tmp_test.append(el)

        test_samples = tmp_test

        classes_names = selected_keys

    # Train: Create dict where a key is the class name, and the value is all the samples that have the same class.

    samples_per_class = dict()
    for cl in classes_names:
        samples_per_class[cl] = [el for el in all_train_samples if el[2] == cl]

    # Split
    splits = []
    print("Shuffling to create splits. May take some time...")
    for i in range(args.nbr_splits):
        for key in samples_per_class.keys():
            for _ in range(1000):
                random.shuffle(samples_per_class[key])
                random.shuffle(samples_per_class[key])
        splits.append(copy.deepcopy(samples_per_class))

    # encode class name into int.
    dict_classes_names = dict()
    for i in range(len(classes_names)):
        dict_classes_names[classes_names[i]] = i

    readme = "csv format:\nrelative path to the image, relative path to the mask, class " \
             "(str). \n You can use the providing encoding of the classes in encoding.yaml"

    # Create the folds.
    def create_folds_of_one_class(lsamps, s_tr, s_vl):
        """
        Create k folds from a list of samples of the same class, each fold contains a train, and valid set with a
        predefined size.

        Note: Samples are expected to be shuffled beforehand.

        :param lsamps: list of paths to samples of the same class.
        :param s_tr: int, number of samples in the train set.
        :param s_vl: int, number of samples in the valid set.
        :return: list_folds: list of k tuples (tr_set, vl_set): where each element is the list (str paths)
                 of the samples of each set: train, and valid, respectively.
        """
        assert len(lsamps) == s_tr + s_vl, "Something wrong with the provided sizes."

        # chunk the data into chunks of size ts (the size of the test set), so we can rotate the test set.
        list_chunks = list(chunk_it(lsamps, s_vl))
        list_folds = []

        for i in range(len(list_chunks)):
            vl_set = list_chunks[i]

            right, left = [], []
            if i < len(list_chunks) - 1:
                right = list_chunks[i + 1:]
            if i > 0:
                left = list_chunks[:i]

            leftoverchunks = right + left

            leftoversamples = []
            for e in leftoverchunks:
                leftoversamples += e

            tr_set = leftoversamples
            list_folds.append((tr_set, vl_set))

        return list_folds

    # Save the folds.
    # Save the folds into *.csv files.
    def dump_fold_into_csv(lsamples, outpath):
        """
        Write a list of RELATIVE paths into a csv file.
        Relative paths allow running the code on any device.
        The absolute path within the device will be determined at the running time.

        csv file format: relative path to the image, relative path to the mask, class (str).

        :param lsamples: list of str of relative paths.
        :param outpath: str, output file name.
        :return:
        """
        with open(outpath, 'w') as fcsv:
            filewriter = csv.writer(fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for im_path, mk_path, cl in lsamples:
                filewriter.writerow([im_path, mk_path, cl])

    def create_one_split(split_i, test_samples, c_split, nbr_folds):
        """
        Create one split of k-folds.

        :param split_i: int, the id of the split.
        :param test_samples: list, list of test samples.
        :param c_split: dict, contains the current split.
        :param nbr_folds: int, number of folds [the k value in k-folds].
        :return:
        """
        l_folds_per_class = []
        for key in c_split.keys():
            # count the number of tr, vl for this current class.
            vl_size = math.ceil(len(c_split[key]) * args.folding["vl"] / 100.)
            tr_size = len(c_split[key]) - vl_size
            # Create the folds.
            list_folds = create_folds_of_one_class(c_split[key], tr_size, vl_size)

            assert len(list_folds) == nbr_folds, "We did not get exactly {} folds, but `{}` .... [ NOT OK]".format(
                nbr_folds,  len(list_folds))

            l_folds_per_class.append(list_folds)

        outd = args.fold_folder
        # Re-arrange the folds.
        for i in range(nbr_folds):
            print("\t Fold: {}".format(i))
            out_fold = join(outd, "split_" + str(split_i) + "/fold_" + str(i))
            if not os.path.exists(out_fold):
                os.makedirs(out_fold)

            # dump the test set
            dump_fold_into_csv(test_samples, join(out_fold, "test_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the train set
            train = []
            for el in l_folds_per_class:
                train += el[i][0]  # 0: tr
            # shuffle
            for t in range(1000):
                random.shuffle(train)

            dump_fold_into_csv(train, join(out_fold, "train_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the valid set
            valid = []
            for el in l_folds_per_class:
                valid += el[i][1]  # 1: vl
            dump_fold_into_csv(valid, join(out_fold, "valid_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the seed
            with open(join(out_fold, "seed.txt"), 'w') as fx:
                fx.write("MYSEED: " + os.environ["MYSEED"])
            # dump the coding.
            with open(join(out_fold, "encoding.yaml"), 'w') as f:
                yaml.dump(dict_classes_names, f)

            with open(join(out_fold, "readme.md"), 'w') as fx:
                fx.write(readme)

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write(readme)
    # dump the coding.
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)
    # Creates the splits
    print("Starting splitting...")
    for i in range(args.nbr_splits):
        print("Split: {}".format(i))
        create_one_split(i, test_samples, splits[i], args.nbr_folds)

    print("All Caltech_UCSD_Birds_200_2011 splitting (`{}`) ended with success .... [OK]".format(args.nbr_splits))


def find_files_pattern(fd_in_, pattern_):
    """
    Find paths to files with pattern within a folder recursively.
    :return:
    """
    assert os.path.exists(fd_in_), "Folder {} does not exist .... [NOT OK]".format(fd_in_)
    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


def create_bin_mask_Oxford_flowers_102(args):
    """
    Create binary masks.
    :param args:
    :return:
    """
    def get_id(pathx, basex):
        """
        Get the id of a sample.
        :param pathx:
        :return:
        """
        rpath = relpath(pathx, basex)
        basen = basename(rpath)
        id = splitext(basen)[0].split('_')[1]
        return id

    baseurl = args.baseurl
    imgs = find_files_pattern(join(baseurl, 'jpg'), '*.jpg')
    bin_fd = join(baseurl, 'segmim_bin')
    if not os.path.exists(bin_fd):
        os.makedirs(bin_fd)
    else:  # End.
        print('Conversion to binary mask has already been done. [OK]')
        return 0

    # Background color [  0   0 254]. (blue)
    print('Start converting the provided masks into binary masks ....')
    for im in tqdm.tqdm(imgs, ncols=80, total=len(imgs)):
        id_im = get_id(im, baseurl)
        mask = join(baseurl, 'segmim', 'segmim_{}.jpg'.format(id_im))
        assert isfile(mask), 'File {} does not exist. Inconsistent logic. .... [NOT OK]'.format(mask)
        msk_in = Image.open(mask, 'r').convert('RGB')
        arr_ = np.array(msk_in)
        arr_[:, :, 0] = 0
        arr_[:, :, 1] = 0
        arr_[:, :, 2] = 254
        blue = Image.fromarray(arr_.astype(np.uint8), mode='RGB')
        dif = ImageChops.subtract(msk_in, blue)
        x_arr = np.array(dif)
        x_arr = np.mean(x_arr, axis=2)
        x_arr = (x_arr != 0).astype(np.uint8)
        img_bin = Image.fromarray(x_arr * 255, mode='L')
        img_bin.save(join(bin_fd, 'segmim_{}.jpg'.format(id_im)), 'JPEG')


def split_Oxford_flowers_102(args):
    """
    Use the provided split: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat

    :param args:
    :return:
    """
    def dump_fold_into_csv(lsamples, outpath):
        """
        Write a list of RELATIVE paths into a csv file.
        Relative paths allow running the code an any device.
        The absolute path within the device will be determined at the running time.

        csv file format: relative path to the image, relative path to the mask, class (str: int).

        :param lsamples: list of str of relative paths.
        :param outpath: str, output file name.
        :return:
        """
        with open(outpath, 'w') as fcsv:
            filewriter = csv.writer(fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for imgx, mkx, clx in lsamples:
                filewriter.writerow([imgx, mkx, clx])
    baseurl = args.baseurl

    # splits
    fin = loadmat(join(baseurl, 'setid.mat'))
    trnid = fin['trnid'].reshape((-1)).astype(np.uint16)
    valid = fin['valid'].reshape((-1)).astype(np.uint16)
    tstid = fin['tstid'].reshape((-1)).astype(np.uint16)

    # labels
    flabels = loadmat(join(baseurl, 'imagelabels.mat'))['labels'].flatten()
    flabels -= 1  # labels are encoded from 1 to 102. We change that to be from 0 to 101.

    # find all the files
    fdimg = join(baseurl, 'jpg')
    tr_set, vl_set, ts_set = [], [], []  # (img, mask, label (int))
    filesin = find_files_pattern(fdimg, '*.jpg')
    lid = []
    for f in filesin:
        rpath = relpath(f, baseurl)
        basen = basename(rpath)
        id = splitext(basen)[0].split('_')[1]
        mask = join(baseurl, 'segmim_bin', 'segmim_{}.jpg'.format(id))
        assert isfile(mask), 'File {} does not exist. Inconsistent logic. .... [NOT OK]'.format(mask)
        rpath_mask = relpath(mask, baseurl)
        id = int(id)  # ids start from 1. Array indexing starts from 0.
        label = int(flabels[id - 1])
        sample = (rpath, rpath_mask, label)
        lid.append(id)
        if id in trnid:
            tr_set.append(sample)
        elif id in valid:
            vl_set.append(sample)
        elif id in tstid:
            ts_set.append(sample)
        else:
            raise ValueError('ID:{} not found in train, valid, test. Inconsistent logic. ....[NOT OK]'.format(id))

    print('Number of samples:\n'
          'Train: {} \n'
          'valid: {} \n'
          'Test: {}\n'
          'Toal: {}'.format(len(tr_set), len(vl_set), len(ts_set), len(tr_set) + len(vl_set) + len(ts_set)))

    dict_classes_names = dict()
    uniquel = np.unique(flabels)
    for i in range(uniquel.size):
        dict_classes_names[str(uniquel[i])] = int(uniquel[i])

    outd = args.fold_folder
    out_fold = join(outd, "split_" + str(0) + "/fold_" + str(0))
    if not os.path.exists(out_fold):
        os.makedirs(out_fold)

    dump_fold_into_csv(tr_set, join(out_fold, "train_s_" + str(0) + "_f_" + str(0) + ".csv"))
    dump_fold_into_csv(vl_set, join(out_fold, "valid_s_" + str(0) + "_f_" + str(0) + ".csv"))
    dump_fold_into_csv(ts_set, join(out_fold, "test_s_" + str(0) + "_f_" + str(0) + ".csv"))

    with open(join(out_fold, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)


# =========================================================================================
#                               RUN
# =========================================================================================


def do_glas():
    """
    GlaS.

    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.set_seed()

    # ===========================

    username = getpass.getuser()
    if username == "xxxx2020":
        baseurl = "xxxx2020/datasets/GlaS-2015/Warwick QU Dataset (Released 2016_07_08)"
    elif username == "sbelharb":
        baseurl = "xxxx2020/datasets/GlaS-2015/Warwick QU Dataset (Released 2016_07_08)"
    else:
        raise ValueError("Cause: anonymization of the code. username `{}` unknown. Set the absolute path to the Caltech-UCSD-Birds-200-2011 dataset. See above for an example .... [NOT OK]".format(username))

    args = {"baseurl": baseurl,
            "folding": {"vl": 20},  # 80 % for train, 20% for validation.
            "dataset": "glas",
            "fold_folder": "folds/glas-test",
            "img_extension": "bmp",
            "nbr_splits": 2  # how many times to perform the k-folds over the available train samples.
            }
    args["nbr_folds"] = math.ceil(100. / args["folding"]["vl"])
    split_valid_glas(Dict2Obj(args))


def do_Caltech_UCSD_Birds_200_2011():
    """
    Caltech-UCSD-Birds-200-2011.

    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.set_seed()

    # ===========================

    username = getpass.getuser()
    if username == "xxxx2020":
        baseurl = "xxxx2020/datasets/Caltech-UCSD-Birds-200-2011"
    elif username == "xxxx2020":
        baseurl = "xxxx2020/datasets/Caltech-UCSD-Birds-200-2011"
    else:
        raise ValueError("Cause: anonymization of the code. username `{}` unknown. Set the absolute path to the Caltech-UCSD-Birds-200-2011 dataset. See above for an example .... [NOT OK]".format(username))

    args = {"baseurl": baseurl,
            "folding": {"vl": 20},  # 80 % for train, 20% for validation.
            "dataset": "Caltech-UCSD-Birds-200-2011",
            "fold_folder": "folds/Caltech-UCSD-Birds-200-2011",
            "img_extension": "bmp",
            "nbr_splits": 2,  # how many times to perform the k-folds over the available train samples.
            "path_encoding": "folds/Caltech-UCSD-Birds-200-2011/encoding-origine.yaml",
            "nbr_classes": None  # Keep only 5 random classes. If you want to use the entire dataset, set this to None.
            }
    args["nbr_folds"] = math.ceil(100. / args["folding"]["vl"])
    split_valid_Caltech_UCSD_Birds_200_2011(Dict2Obj(args))


def do_Oxford_flowers_102():
    """
    Oxford-flowers-102.
    The train/valid/test sets are already provided.

    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.set_seed()

    # ===========================

    username = getpass.getuser()
    if username == "xxxx2020":
        baseurl = "xxxxx2020/datasets/Oxford-flowers-102"
    elif username == "xxxx2020":
        baseurl = "xxxx2020/datasets/Oxford-flowers-102"
    else:
        raise ValueError("Cause: anonymization of the code. username `{}` unknown. Set the absolute path to the Caltech-UCSD-Birds-200-2011 dataset. See above for an example .... [NOT OK]".format(username))

    args = {"baseurl": baseurl,
            "dataset": "Oxford-flowers-102",
            "fold_folder": "folds/Oxford-flowers-102",
            "img_extension": "jpg",
            "path_encoding": "folds/Oxford-flowers-102/encoding-origine.yaml"
            }
    # Convert masks into binary masks.
    create_bin_mask_Oxford_flowers_102(Dict2Obj(args))
    reproducibility.set_seed()
    split_Oxford_flowers_102(Dict2Obj(args))

    # Find min max size.
    def find_stats(argsx):
        """

        :param argsx:
        :return:
        """
        minh, maxh, minw, maxw = None, None, None, None
        baseurl = argsx.baseurl
        fin = find_files_pattern(join(baseurl, 'jpg'), '*.jpg')
        print("Computing stats from {} dataset ...".format(argsx.dataset))
        for f in tqdm.tqdm(fin, ncols=80, total=len(fin)):
            w, h = Image.open(f, 'r').convert('RGB').size
            if minh is None:
                minh = h
                maxh = h
                minw = w
                maxw = w
            else:
                minh = min(minh, h)
                maxh = max(maxh, h)
                minw = min(minw, w)
                maxw = max(maxw, w)

        print('Stats {}:\n'
              'min h: {} \n'
              'max h: {} \n'
              'min w: {} \n'
              'max w: {} \n'.format(argsx.dataset, minh, maxh, minw, maxw))

    find_stats(Dict2Obj(args))



if __name__ == "__main__":
    # ============== CREATE FOLDS OF GlaS DATASET
    do_glas()

    # ============== CREATE FOLDS OF Caltech-UCSD-Birds-200-2011 DATASET
    # do_Caltech_UCSD_Birds_200_2011()

    # ============== CREATE FOLDS OF Oxford-flowers-102 DATASET
    # do_Oxford_flowers_102()
