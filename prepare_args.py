import random
from random import shuffle
import os
import json


def get_index_of_char(my_string, char):
    """Returns all indices of all appearances of char in str

    :param file_name:
    :param char:
    :return:
    """
    return [x for x, v in enumerate(my_string) if v == char]


def _save_dictionary_to_json(my_dict, json_path, file_name):
    with open(os.path.join(json_path, file_name), 'w') as json_file:
        json.dump(my_dict, json_file)


def save_download_args(source_dir, results_dir):
    """This function prepares and saves (as dictionary) all arguments that are needed
    to generate lmdb files from the csv files.
    under "left out" we save the instance that was used for testing from each object.
    Currently it is producing 3 couples of train-test args:
    full (including all views); half (including every second view); third (every third view).
    The arguments that are saved are:
    A list of files (views); results dir; name of the db

    :param source_dir:
    :param results_dir:
    :return:
    """

    objects_dict = {}  # mapping of objects and which instance is for testing

    training_list_objects = []
    testing_list_objects = []

    training_list_objects_half = []
    testing_list_objects_half = []

    training_list_objects_third = []
    testing_list_objects_third = []

    training_list_objects_fifth = []
    testing_list_objects_fifth = []

    for dirName, subdirList, fileList in os.walk(source_dir):
        for file_name in fileList:

            index_ = get_index_of_char(file_name, "_")

            # dir containing the fpfh files have some dirs and files that we should ignore.
            # Super hacky and ugly :(
            if "extraData" in dirName or file_name.startswith(".") or not index_:
                continue

            object_name = file_name[index_[0]]
            instance_number = file_name[index_[0] + 1:index_[1]]
            video_number = file_name[index_[1] + 1:index_[2]]
            frame_number = file_name[index_[2] + 1:index_[3]]

            if object_name not in objects_dict.keys():
                objects_dict[object_name] = random.randint(1, 4)

            if int(instance_number) == objects_dict[object_name]:
                testing_list_objects.append(os.path.join(dirName, file_name))
                if int(frame_number) % 2 == 0:
                    testing_list_objects_half.append((os.path.join(dirName, file_name)))
                if int(frame_number) % 3 == 0:
                    testing_list_objects_third.append((os.path.join(dirName, file_name)))
                if int(frame_number) % 5 == 0:
                    testing_list_objects_fifth.append((os.path.join(dirName, file_name)))
            else:
                training_list_objects.append(os.path.join(dirName, file_name))
                if int(frame_number) % 2 == 0:
                    training_list_objects_half.append((os.path.join(dirName, file_name)))
                if int(frame_number) % 3 == 0:
                    training_list_objects_third.append((os.path.join(dirName, file_name)))
                if int(frame_number) % 5 == 0:
                    training_list_objects_fifth.append((os.path.join(dirName, file_name)))

    print "number of items in complete training and testing sets"
    print len(training_list_objects), len(testing_list_objects)

    print "number of items in training and testing sets - every second view"
    print len(training_list_objects_half), len(testing_list_objects_half)

    print "number of items in training and testing sets - every third view"
    print len(training_list_objects_third), len(testing_list_objects_third)

    print "number of items in training and testing sets - every fifth view"
    print len(training_list_objects_fifth), len(testing_list_objects_fifth)

    shuffle(training_list_objects)
    shuffle(training_list_objects_half)
    shuffle(training_list_objects_third)
    shuffle(training_list_objects_fifth)

    args = {"args": [{"files_list": training_list_objects, "results_dir": results_dir,
                      "train_test": "training_full"},
                     {"files_list": testing_list_objects, "results_dir": results_dir,
                      "train_test": "testing_full"},
                     {"files_list": training_list_objects_half, "results_dir": results_dir,
                      "train_test": "training_half"},
                     {"files_list": testing_list_objects_half, "results_dir": results_dir,
                      "train_test": "testing_half"},
                     {"files_list": training_list_objects_third, "results_dir": results_dir,
                      "train_test": "training_third"},
                     {"files_list": testing_list_objects_third, "results_dir": results_dir,
                      "train_test": "testing_third"},
                     {"files_list": training_list_objects_fifth, "results_dir": results_dir,
                      "train_test": "training_fifth"},
                     {"files_list": testing_list_objects_fifth, "results_dir": results_dir,
                      "train_test": "testing_fifth"}],
            "left_out": objects_dict}

    _save_dictionary_to_json(args, results_dir, "args_test.json")


def main():
    save_download_args("/mnt/scratch/noa/pclproj/fpfh", "/mnt/scratch/noa/pclproj/results")

if __name__ == "__main__":
    main()
