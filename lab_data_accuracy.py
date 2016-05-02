import numpy as np
import caffe
import json
import lmdb
import os
from scipy.stats import ttest_rel as ttest


MAX_WIDTH = 227
MAX_HEIGHT = 168
CHANNELS_NUMBER = 36

def get_coordinates(coordinates_file_name):
	''' This function reads a csv file to a numpy array and moves the 
	pixel coordinates to the top left corner.
	imX: index of imX in a row
	imY: index of imY in a row
	'''

	coordinates = np.genfromtxt(coordinates_file_name, delimiter=" ", dtype="int")
	coordinates[:, 0] -= coordinates[:, 0].min()
	coordinates[:, 1] -= coordinates[:, 1].min()
	
	return coordinates

def _prepare_data(coordinates, view_file):
	'''This function takes a 2D array (points X (channels)) 
	and converts it to 3D (channels X max_height X max_width)
	'''
	hist_rgb_mat = np.genfromtxt(view_file, delimiter=" ", dtype="float32")
	data_matrix = np.zeros((CHANNELS_NUMBER, MAX_HEIGHT, MAX_WIDTH))
	for coor, hist_rgb in zip(coordinates, hist_rgb_mat):
		i = coor[1]
		j = coor[0]
		data_matrix[:, i, j] = hist_rgb

	return data_matrix

def get_mean_array(mean_binaryproto_path):
	blob_t = caffe.proto.caffe_pb2.BlobProto()
	data_t = open(mean_binaryproto_path , 'rb' ).read()
	blob_t.ParseFromString(data_t)
	arr_t = np.array(caffe.io.blobproto_to_array(blob_t))[0,:,:,:]
	
	return arr_t

def get_true_label(view_file, labels_dict):
	for key in labels_dict.keys():
		if os.path.basename(key) in view_file:
			return int(labels_dict[key])
	
	print 'did not find label for: ' + view_file
	return

def get_prediction(data_3d, net):
	net.blobs['data'].data[...] =  data_3d
	out = net.forward()
	
	return out['prob'].argmax()

def get_truth_predictions(deploy_prototxt, model_prototxt, files_list, 
				 coordinates_files_list, mean_binaryproto_path, labels_dict):
	
	net = caffe.Net(deploy_prototxt, model_prototxt, caffe.TEST)
	
	if not type(mean_binaryproto_path) is list:
		mean_array = get_mean_array(mean_binaryproto_path)
	else:
		mean_hist = get_mean_array(mean_binaryproto_path[0])
		mean_rgb = get_mean_array(mean_binaryproto_path[1])
	
	if 'rgb/' in model_prototxt:
		net.blobs['data'].reshape(1, 3,168,227)
	elif 'rgb_hist' in model_prototxt:
		net.blobs['data'].reshape(1, 36,168,227)
	else:
		net.blobs['data'].reshape(1, 33,168,227)
	
	true_labels = []
	predicted_labels = []
	
	for view_file, coor_file in zip(files_list, coordinates_files_list):
		coor = get_coordinates(coor_file)
		try:
			data_3d = _prepare_data(coor, view_file)
		except IndexError:
			print view_file
			continue
		
		if 'rgb/' in model_prototxt:
			data_3d = (data_3d[33:,:,:] - mean_array)
		elif '/hist/' in model_prototxt:
			data_3d = data_3d[:33,:,:] - mean_array
		else:
			hist = data_3d[:33,:,:] - mean_hist
			rgb = data_3d[33:,:,:] - mean_rgb
			data_3d = np.concatenate((rgb,hist))
			
		
		
		true_labels.append(get_true_label(view_file, labels_dict))
		predicted_labels.append(get_prediction(data_3d, net))
	
	return true_labels, predicted_labels

def main():

	with open('/mnt/scratch/noa/pclproj/results/args_lab_data.json') as data_file:
		args = json.load(data_file)

	with open('/mnt/scratch/noa/pclproj/results/labels_objects_dict.json') as data_file:
		labels = json.load(data_file)

	#tot_acc = np.array([0.0, 0.0, 0.0])
	tot_acc = [[], [], []]
	dates_list = ['1601', '1801', '2101', '2701', '0302', '0702', '1202']
	for date_folder in dates_list:
		print date_folder
		true_labels_rgb, predictions_rgb = get_truth_predictions('/home/noa/pcl_proj/experiments/cifar10/every_fifth_view/'+
													  date_folder+'/rgb/deploy.prototxt',
													'/home/noa/pcl_proj/experiments/cifar10/every_fifth_view/' 
													+date_folder+'/rgb/snapshots/_iter_200000.caffemodel',
													args["views_files"], args["coors_files"], 
													'/home/noa/pcl_proj/experiments/mean_image_files/'
													+date_folder+'/mean_image_training_fifth_rgb.binaryproto',
													labels)
		true_labels_hist, predictions_hist = get_truth_predictions('/home/noa/pcl_proj/experiments/cifar10/every_fifth_view/'+
											  date_folder+'/hist/deploy.prototxt',
											'/home/noa/pcl_proj/experiments/cifar10/every_fifth_view/' 
											+date_folder+'/hist/snapshots/_iter_200000.caffemodel',
											args["views_files"], args["coors_files"], 
											'/home/noa/pcl_proj/experiments/mean_image_files/'
											+date_folder+'/mean_image_training_fifth_hist.binaryproto',
											labels)
		
		true_labels_rgb_hist, predictions_rgb_hist = get_truth_predictions('/home/noa/pcl_proj/experiments/cifar10/every_fifth_view/'+
											  date_folder+'/rgb_hist/deploy.prototxt',
											'/home/noa/pcl_proj/experiments/cifar10/every_fifth_view/' 
											+date_folder+'/rgb_hist/snapshots/_iter_200000.caffemodel',
											args["views_files"], args["coors_files"], 
											['/home/noa/pcl_proj/experiments/mean_image_files/'
											+date_folder+'/mean_image_training_fifth_hist.binaryproto', '/home/noa/pcl_proj/experiments/mean_image_files/'
											+date_folder+'/mean_image_training_fifth_rgb.binaryproto'],
											labels)
		acc = [0.0, 0.0, 0.0]
		for i in range(len(predictions_rgb)):
			if true_labels_rgb[i] == predictions_rgb[i]:
				acc[0] += 1.0
			if true_labels_hist[i] == predictions_hist[i]:
				acc[1] += 1.0
			if true_labels_rgb_hist[i] == predictions_rgb_hist[i]:
				acc[2] += 1.0
		tot_acc[0].append(acc[0] / len(predictions_rgb))
		tot_acc[1].append(acc[1] / len(predictions_hist))
		tot_acc[2].append(acc[2] / len(predictions_rgb_hist))		
		print acc
	print tot_acc
	print 'ttests: '
	statistics_0, p_0 = ttest(np.array(tot_acc[0]), np.array(tot_acc[1]))
	sttistics_1, p_1 = ttest(np.array(tot_acc[1]), np.array(tot_acc[2]))
	sttistics_2, p_2 = ttest(np.array(tot_acc[0]), np.array(tot_acc[2]))


	print "final: "
	print 'rgb:'
	print np.array(tot_acc[0]).mean()
	print np.array(tot_acc[0]).std()
	print 'hist:'
	print np.array(tot_acc[1]).mean()
	print np.array(tot_acc[1]).std()
	print 'rgb_hist:'
	print np.array(tot_acc[2]).mean()
	print np.array(tot_acc[2]).std()

	print 'rgb vs. hist'
	print p_0
	print 'hist vs. rgb_hist'
	print p_1
	print 'rgb vs. rgb_hist'
	print p_2



if __name__ == "__main__":
	main()
