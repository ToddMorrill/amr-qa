"""This module downloads the necessary data artifacts.

Examples:
    $ python download.py \
        --save-dir /home/iron-man/Documents/data/amr-qa
"""
import argparse
import json
import os

import requests


def get_filepath(save_dir, url):
    filename = url.split('/')[-1]
    return os.path.join(save_dir, filename)


def download_save(url, save_dir):
    response = requests.get(url)
    save_filepath = get_filepath(save_dir, response.url)
    with open(save_filepath, 'w') as f:
        json.dump(response.json(), f)


def main(args):
    # ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # files to be downloaded
    qald_train = 'https://raw.githubusercontent.com/ag-sc/QALD/master/9/data/qald-9-train-multilingual.json'
    qald_test = 'https://raw.githubusercontent.com/ag-sc/QALD/master/9/data/qald-9-test-multilingual.json'
    download_save(url=qald_train, save_dir=args.save_dir)
    download_save(url=qald_test, save_dir=args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save-dir',
        help='The directory where the data will be saved. If it does not '
        'exist, it will be created.')
    args = parser.parse_args()
    main(args)