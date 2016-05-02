import os
import json

import numpy as np
import caffe
import click

imX = 36
imY = 37
channels_number = 36
max_height = 168
max_width = 227


def read_data_from_file(file_name):
    ''' This function reads a csv file to a numpy array and moves the
    pixel coordinates to the top left corner.
    imX: index of imX in a row
    imY: index of imY in a row
    '''

    data = np.genfromtxt(file_name, delimiter=" ")
    data[:, imX] -= min(data[:, imX])
    data[:, imY] -= min(data[:, imY])

    return data


def prepare_data(data_2d):
    '''This function takes a 2D array (points X (channels+2))
    and converts it to 3D (channels X max_height X max_width)
    imX: index of imX in a row
    imY: index of imY in a row
    '''
    data_matrix = np.zeros((channels_number, max_height, max_width))
    for row in data_2d:
        i = row[imY]
        j = row[imX]
        data_matrix[:, i, j] = row[:channels_number]

    return data_matrix


def calculate_mean_image(files_list, hist):

    dim = 33 if hist else 3
    
    mean_image = np.zeros((dim, max_height, max_width))
    num_images = 0

    for file_name in files_list:
        if hist:
            data_3d = prepare_data(read_data_from_file(file_name))[:33]
        else:
            data_3d = prepare_data(read_data_from_file(file_name))[33:]
        mean_image += data_3d
        num_images += 1

    return mean_image, num_images


def save_prototype_files(args_json_path, dir_name, hist, index):
    with open(args_json_path) as data_file:
        args = json.load(data_file)

    arg = args['args'][index]
    files_list = arg['files_list']
    specification_string = arg['train_test']

    mean_image, num_images = calculate_mean_image(files_list, hist)
    mean_image /= num_images

    if len(mean_image.shape) == 3:
        mean_image = np.expand_dims(mean_image, axis=0)

    print mean_image.shape

    ext = '_hist.binaryproto' if hist else '_rgb.binaryproto'
    file_name = 'mean_image_' + specification_string + ext
    file_path = os.path.join(dir_name, file_name)

    blob_proto = caffe.io.array_to_blobproto(mean_image)
    with open(file_path, 'w') as binaryproto_file:
        binaryproto_file.write(blob_proto.SerializeToString())

    print 'wrote file: ' + file_path

def generate_mean_files(arguments_file, results_path):
    save_prototype_files(arguments_file, results_path, False, 6)
    save_prototype_files(arguments_file, results_path, True, 6)

