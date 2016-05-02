import click

from prepare_args_all_instances import prepare_args
from generate_labels import generate_labels
from generate_mean_image_binaryprotos import generate_mean_files
from read_csv_seperated_rgb_histograms import generate_lmdbs



@click.command()
@click.option('--source_dir', type=click.STRING,
			  default='/mnt/scratch/noa/pclproj/fpfh',
			  help='directory containing fpfh csv files')
@click.option('--lmdb_dir', type=click.STRING,
			  default='/mnt/scratch/noa/pclproj/results/',
			  help='path to save the json file')
@click.option('--mean_dir', type=click.STRING,
			  default='/home/noa/pcl_proj/experiments/mean_images',
			  help='dictionary to save the json file')
@click.option('--labels_dict', type=click.STRING,
              default='/mnt/scratch/noa/pclproj/results/labels_objects_dict.json',
              help='dictionary mapping between objects and labels')
def main(source_dir, lmdb_dir, mean_dir, labels_dict):
	print '============================================================================'
	print 'beginning generating argument file'
	args_file_path = prepare_args(source_dir, lmdb_dir)
	print 'Done generating arguments file'
	print '============================================================================'
	print 'Begining to generate labels'
	generate_labels(args_file_path, labels_dict, lmdb_dir)
	print 'Done generating labels'
	print '============================================================================'
	print 'Begining to generate mean files'
	generate_mean_files(args_file_path, mean_dir)
	print 'Done generating mean files'
	print '============================================================================'
	print 'Beginning to generate lmdb files'
	generate_lmdbs(args_file_path)
	print 'Done generating lmdb files'


if __name__ == "__main__":
    main()