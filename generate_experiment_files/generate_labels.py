import json
import os

import click
import caffe
import lmdb
import numpy as np

NUM_IDX_DIGITS = 10
IDX_FMT = '{:0>%d' % NUM_IDX_DIGITS + 'd}'

def _save_labels_2_lmdb(labels, db_path, lut = None):
    """This function saves list of integers (labels) to lmdb.
    credit: Youssef Kashef

    :param labels:
    :param db_path:
    :return:
    """
    db = lmdb.open(db_path, map_size=int(1e12))

    with db.begin(write=True) as in_txn:

        if not hasattr(labels, '__iter__'):
            labels = np.array([labels])

        for idx, x in enumerate(labels):

            if not hasattr(x, '__iter__'):
                content_field = np.array([x])
            else:
                content_field = np.array(x)

            # validate these are scalars
            if content_field.size != 1:
                raise AttributeError("Unexpected shape for scalar at i=%d (%s)"
                                     % (idx, str(content_field.shape)))

            # guarantee shape (1,1,1)
            while len(content_field.shape) < 3:
                content_field = np.expand_dims(content_field, axis=0)

            content_field = content_field.astype(int)

            if lut is not None:
                content_field = lut(content_field)

            dat = caffe.io.array_to_datum(content_field)
            in_txn.put(IDX_FMT.format(idx), dat.SerializeToString())

    db.close()

def generate_labels_per_index(args, labels, results_path, index):
    files_list = args["args"][index]['files_list']

    labels_list = []
    for file_name in files_list:
        for key in labels.keys():
            if key in file_name:
                labels_list.append(int(labels[key]))
                break

    db_path = os.path.join(results_path, "labels_" + args["args"][index]["train_test"] + '.lmdb')
    print db_path
    _save_labels_2_lmdb(labels_list, db_path)

def generate_labels(arguments_file, labels_dict, results_path):
    with open(arguments_file) as data_file:
        args = json.load(data_file)

    with open(labels_dict) as data_file:
        labels = json.load(data_file)

    generate_labels_per_index(args, labels, results_path, 6)
    generate_labels_per_index(args, labels, results_path, 7)
    