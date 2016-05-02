import os
import random
from random import shuffle
import multiprocessing

import numpy as np
import json
import lmdb
import caffe
import click

NUM_IDX_DIGITS = 10
IDX_FMT = '{:0>%d' % NUM_IDX_DIGITS + 'd}'

IM_X = 36
IM_Y = 37
MAX_WIDTH = 227
MAX_HEIGHT = 168
CHANNELS_NUMBER = 36

class LmdbGenerator(object):
    def __init__(self, im_x=IM_X, im_y=IM_Y, max_width=MAX_WIDTH, max_height=MAX_HEIGHT,
                 channels_number=CHANNELS_NUMBER):

        # data settings
        self.imX = im_x
        self.imY = im_y
        self.max_width = max_width
        self.max_height = max_height
        self.channels_number = channels_number

    def _read_data_from_file(self, file_name):
        ''' This function reads a csv file to a numpy array and moves the 
        pixel coordinates to the top left corner.
        imX: index of imX in a row
        imY: index of imY in a row
        '''

        data = np.genfromtxt(file_name, delimiter=" ", dtype="float32")
        data[:, self.imX] -= min(data[:, self.imX])
        data[:, self.imY] -= min(data[:, self.imY])

        return data

    def _prepare_data(self, data_2d):
        '''This function takes a 2D array (points X (channels+2)) 
        and converts it to 3D (channels X max_height X max_width)
        imX: index of imX in a row
        imY: index of imY in a row
        '''
        data_matrix = np.zeros((self.channels_number, self.max_height, self.max_width))
        for row in data_2d:
            i = row[self.imY]
            j = row[self.imX]
            data_matrix[:, i, j] = row[:self.channels_number]

        return data_matrix

    def _insert_data_to_lmdb(self, db_dir, files_list, train_test, hist):
        '''
        Generate LMDB file from list of a list of csv files
        credit: Youssef Kashef
        '''

        ext = '_hist_all_instances.lmdb' if hist else '_rgb_all_instances.lmdb'
        db_path = os.path.join(db_dir, "data_" + train_test + ext)
        db = lmdb.open(db_path, map_size=int(1e12))


        with db.begin(write=True) as in_txn_rgb:
            idx = 0

            for fname in files_list:

                print fname
                
                data_2d = self._read_data_from_file(fname)
                data_3d = self._prepare_data(data_2d)

                if hist:
                    dat = caffe.io.array_to_datum(data_3d[:33])
                else:
                    dat = caffe.io.array_to_datum(data_3d[33:])
                in_txn_rgb.put(IDX_FMT.format(idx), dat.SerializeToString())
            
                idx += 1
                print idx

                

        print 'I am done!!!!!!'
        db.close()

        return 0


    #@profile
    def generate_lmdb(self, hist, files_list, results_dir, train_test):
        self._insert_data_to_lmdb(results_dir, files_list, train_test, hist)

        return 0


def generate_single_lmdb(data_args, hist):
    lmdb_generator = LmdbGenerator()

    lmdb_generator.generate_lmdb(hist, files_list=data_args["files_list"],
                                 results_dir=data_args["results_dir"],
                                 train_test=data_args["train_test"])


@click.command()
@click.option('--arguments_file', type=click.STRING,
              default='/mnt/scratch/noa/pclproj/results/args_rgbd.json',
              help='arguments list')
@click.option('--hist', type=click.BOOL, default=False,
              help='if true generate histograms lmdb, otherwise rgb.')
@click.option('--index', type=click.INT,
              help='index in the args list')
def main(arguments_file, hist, index):
    with open(arguments_file) as data_file:
        data_args = json.load(data_file)


    generate_single_lmdb(data_args["args"][index], hist)


if __name__ == "__main__":
    main()

