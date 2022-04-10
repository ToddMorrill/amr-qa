"""This module downloads the necessary data artifacts.

Examples:
    $ python download.py \
        --save-dir /home/iron-man/Documents/data/amr-qa
"""
import argparse
from ast import mod
import json
import os
import tarfile
import io
import shutil

import amrlib
import requests


def get_filepath(save_dir, url):
    filename = url.split('/')[-1]
    return os.path.join(save_dir, filename)


def download_save(url, save_dir):
    response = requests.get(url)
    save_filepath = get_filepath(save_dir, response.url)
    with open(save_filepath, 'w') as f:
        json.dump(response.json(), f)

def download_amr_model(url, save_dir):
    response = requests.get(url, stream=True)
    tar = tarfile.open(fileobj=io.BytesIO(response.content))
    tar.extractall(save_dir)
    tar.close()

def exists(save_dir, url, type='tar'):
    filepath = get_filepath(save_dir, url)
    if type == 'tar':
        # remove file extension
        path_check = filepath[:filepath.index('.')]
    elif type == 'json':
        path_check = filepath
    elif type == 'amr':
        dest_dir = os.path.split(amrlib.__file__)[0]
        data_dir = os.path.join(dest_dir, 'data')
        tar_file = url.split('/')[-1]
        folder_name = tar_file[:tar_file.index('.')]
        model_type = 'model_stog' if 'parse' in folder_name else 'model_gtos'
        path_check = os.path.join(data_dir, model_type)
    else:
        raise Exception('Argument type must be one of {tar, json, amr}.')
    return os.path.exists(path_check)


def main(args):
    # ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # files to be downloaded
    qald_train = 'https://raw.githubusercontent.com/ag-sc/QALD/master/9/data/qald-9-train-multilingual.json'
    qald_test = 'https://raw.githubusercontent.com/ag-sc/QALD/master/9/data/qald-9-test-multilingual.json'
    for url in [qald_train, qald_test]:
        # check if cached already
        if not exists(save_dir=args.save_dir, url=url, type='json'):
            print(f'Downloading {url.split("/")[-1]}.')
            download_save(url=url, save_dir=args.save_dir)

    amr_stog = 'https://github.com/bjascob/amrlib-models/releases/download/parse_xfm_bart_large-v0_1_0/model_parse_xfm_bart_large-v0_1_0.tar.gz'
    amr_gtos = 'https://github.com/bjascob/amrlib-models/releases/download/model_generate_t5wtense-v0_1_0/model_generate_t5wtense-v0_1_0.tar.gz'
    for url in [amr_stog, amr_gtos]:
        # check if cached already
        if not exists(save_dir=args.save_dir, url=url, type='amr'):
            print(f'Downloading AMR model {url.split("/")[-1]}. This may take some time.')
            download_amr_model(url=url, save_dir=args.save_dir)

            # mv downloaded models to amrlib/data directory
            # https://amrlib.readthedocs.io/en/latest/install/
            dest_dir = os.path.split(amrlib.__file__)[0]
            data_dir = os.path.join(dest_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            tar_file = url.split('/')[-1]
            folder_name = tar_file[:tar_file.index('.')]
            src = os.path.join(args.save_dir, folder_name)
            model_type = 'model_stog' if 'parse' in folder_name else 'model_gtos'
            dst = os.path.join(data_dir, model_type)
            path = shutil.move(src, dst)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save-dir',
        help='The directory where the data will be saved. If it does not '
        'exist, it will be created.')
    args = parser.parse_args()
    main(args)