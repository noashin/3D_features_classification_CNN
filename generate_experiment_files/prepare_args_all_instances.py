import random
from random import shuffle
import os
import json
import click

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


def count_views_for_instance(source_dir):
	count_dict = {}
	
	for dirName, subdirList, fileList in os.walk(source_dir):
		count_dict[dirName] = 0
		for file_name in fileList:
			index_ = get_index_of_char(file_name, "_")
			if "extraData" in dirName or file_name.startswith(".") or not index_:
				continue
			
			count_dict[dirName] += 1
			
	return count_dict

def instances_paths_dict(source_dir):
	'''This function gets a path to all the csv files
	and returns a dictionary where key is 
	'''
	instances_dict = {}
		
	for dirName, subdirList, fileList in os.walk(source_dir):
		if "extraData" in dirName or dirName[-1] == 'h':
			continue
		instances_dict[dirName] = []
		for file_name in fileList:
			index_ = get_index_of_char(file_name, "_")
			if file_name.startswith(".") or not index_:
				continue
			
			instances_dict[dirName].append(file_name)
	return instances_dict


def get_instances_numbers(instances_dict):
	"""This function gets a dictionary of all the views and returns a dictionary
	with objects as keys and a list of the indices of the instances of the object
	for value.
	"""
	num_instances_dict = {}
	for key in instances_dict.keys():
		index_ = get_index_of_char(key, "_")
		instance_number = int(key[index_[0]+1:])
		object_name = key[:index_[0]]
		if object_name not in num_instances_dict.keys():
			num_instances_dict[object_name] = []
		if instance_number not in num_instances_dict[object_name]:
			num_instances_dict[object_name].append(instance_number)
	return num_instances_dict

def chose_which_to_leave(views_dict, instances_num_dict):
	''' This function gets the views dictionary and choses randomly
	for each object which instance to use for testing.
	returns a dictionary with objects as keys and the number of the 
	instance that was left out as value.
	'''
	left_out_objects = {}
	for key in views_dict.keys():
		index_ = get_index_of_char(key, "_")
		instance_number = int(key[index_[0]+1:])
		object_name = key[:index_[0]]
		if object_name not in left_out_objects.keys():
			inst_num = random.randint(0, len(instances_num_dict[object_name]) - 1)
			left_out_objects[object_name] = instances_num_dict[object_name][inst_num]

	return left_out_objects
				
	

def leave_1_out(views_dict, left_out_objects):
	'''
	views_dict: dictionary of frames. key is a folder (objectname_instancenumber)
	the function iterates over the views_dict and seperates it to train and test data
	following leave one out paradigm.
	'''
	
	training_files={}
	testing_files={}
	for key in views_dict:
		object_name = key[:-2]
		instance_number = int(key[-1])
		
		if instance_number == left_out_objects[object_name]:
			testing_files[key] = []
			for file_name in views_dict[key]:
				testing_files[key].append(file_name)
		else:
			training_files[key] = []
			for file_name in views_dict[key]:
				training_files[key].append(file_name)

	return training_files, testing_files
				
			
def every_n_view(views_dict, n):
	''' takes oly the nth view for each instances and returns
	a new dictionary
	'''
	nth_view_dict = {}
	
	for key in views_dict:
		nth_view_dict[key] = []
		for file_name in views_dict[key]:
			index_ = get_index_of_char(file_name, "_")
			instance_number = file_name[index_[0] + 1:index_[1]]
			frame_number = file_name[index_[2] + 1:index_[3]]

			if int(frame_number) % n == 0:
				nth_view_dict[key].append(file_name)          
	return nth_view_dict


def arrange_dict_according_to_objects(views_dict):
	'''This function gets the views_dictionary were the keys are instances_dict
	and return a dictionary with the objects as keys
	'''
	objects_dict = {}
	for key in views_dict.keys():
		index_ = get_index_of_char(key, "_")
		object_path = key[:index_[0]]
		if object_path not in objects_dict.keys():
			objects_dict[object_path] = []
		objects_dict[object_path] += views_dict[key]
	
	return objects_dict


def shuffle_and_make_even(views_dict):
	''' This function gets a dictionary of all the views 
	and return a shuffled list containing same number of views for each object
	'''
	minimum_views = 1000
	minimum_views_key = ''

	objects_dict = arrange_dict_according_to_objects(views_dict)
	
	num_views_per_object = [len(objects_dict[key]) for key in objects_dict.keys()]

	min_num_views = min(num_views_per_object)

	files_list = []
	for key in objects_dict.keys():
		shuffle(objects_dict[key])
		for file_name in objects_dict[key][:min_num_views]:
			index_ = get_index_of_char(file_name, "_")
			instance_num = file_name[index_[0]:index_[1]]
			files_list.append(os.path.join(key + instance_num, file_name))
	
	shuffle(files_list)

	return files_list

def prepare_test_train(instances_dict, n, left_out_objects):
	every_nth_view = every_n_view(instances_dict, n)
	training, testing = leave_1_out(every_nth_view, left_out_objects)
	training_equal = shuffle_and_make_even(training)
	testing_equal = shuffle_and_make_even(testing)
	
	return training_equal, testing_equal

def save_download_args(source_dir, results_dir, file_name):
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
	
	
	instances_dict = instances_paths_dict("/mnt/scratch/noa/pclproj/fpfh")
	
	left_out_objects = chose_which_to_leave(instances_dict, get_instances_numbers(instances_dict))

	training_list_objects, testing_list_objects = \
		prepare_test_train(instances_dict, 1, left_out_objects)

	training_list_objects_half, testing_list_objects_half = \
		prepare_test_train(instances_dict, 2, left_out_objects)

	training_list_objects_third , testing_list_objects_third = \
		prepare_test_train(instances_dict, 3, left_out_objects)

	training_list_objects_fifth, testing_list_objects_fifth = \
		prepare_test_train(instances_dict, 5, left_out_objects)
	
	print "number of items in complete training and testing sets"
	print len(training_list_objects), len(testing_list_objects)

	print "number of items in training and testing sets - every second view"
	print len(training_list_objects_half), len(testing_list_objects_half)

	print "number of items in training and testing sets - every third view"
	print len(training_list_objects_third), len(testing_list_objects_third)

	print "number of items in training, testing sets and validation - every fifth view"
	print len(training_list_objects_fifth), len(testing_list_objects_fifth)

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
					  "train_test": "testing_fifth"},
					],
			"left_out": left_out_objects}

	_save_dictionary_to_json(args, results_dir, file_name)
	return os.path.join(results_dir, file_name)



def prepare_args(source_dir, results_dir):
	file_name = 'args.json'
	return save_download_args(source_dir, results_dir, file_name)