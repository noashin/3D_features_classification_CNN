import os
import json

import click


def generate_prototxt(replacement_dict, template_file_name, new_file_name):
    """

    :param replacement_dict: {'training_data_mean': ,
                                'training_data': , 'training_labels': ,
                                'testing_data_mean': , 'testing_data': ,
                                'testing_labels': }
    :return:
    """


    with open(new_file_name, 'w') as new_file:
        with open(template_file_name) as template_file:
            for line in template_file:
                for key in replacement_dict:
                    line = line.replace(key, replacement_dict[key])
                new_file.write(line)


def edit_solver(test_iter, test_interval, network_file, snapshot_prefix,
                template_file, new_file):
    replacement_dict = {'testiter': test_iter,
                        'testinterval': test_interval,
                        'network': network_file,
                        'snapshotprefix': snapshot_prefix}

    generate_prototxt(replacement_dict, template_file, new_file)


@click.command()
@click.option('--network_prototxt_dict', type=click.STRING,
              default=None)
@click.option('--solver_prototxt_dict', type=click.STRING,
              default=None)
@click.option('--network_template_file', type=click.STRING,
              default=None)
@click.option('--solver_template_file', type=click.STRING,
              default=None)
@click.option('--new_network_file', type=click.STRING,
              default=None)
@click.option('--new_solver_file', type=click.STRING,
              default=None)
def main(network_prototxt_dict, solver_prototxt_dict, network_template_file, solver_template_file,
         new_solver_file, new_network_file):

    if network_prototxt_dict:
        with open(network_prototxt_dict) as data_file:
            network_replacement_dict = json.load(data_file)
        generate_prototxt(network_replacement_dict, network_template_file, new_network_file)

    if solver_prototxt_dict:
        with open(solver_prototxt_dict) as data_file:
            solver_replacement_dict = json.load(data_file)
        generate_prototxt(solver_replacement_dict, solver_template_file, new_solver_file)


if __name__ == "__main__":
    main()
