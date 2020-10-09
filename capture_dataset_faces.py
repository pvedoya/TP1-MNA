from lib.face_detection import face_recognition
import argparse
from pathlib import Path
import configparser

config = configparser.ConfigParser()
config.read('configuration.ini')
img_height = config.getint('IMAGES_DATA', 'HEIGHT')
img_width = config.getint('IMAGES_DATA', 'WIDTH')
img_path = config.get('SETTINGS', 'PHOTO_SET')


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help='Name of the person taking the pictures', required=True)
parser.add_argument('-d', '--dir', help='Path of the dataset directory', required=False, default=img_path)
parser.add_argument('-p', '--pictures', help='Amount of pictures to be taken', required=False, default=10)
parser.add_argument('-iw', '--width', help='Desired image width', required=False, default=img_width)
parser.add_argument('-ih', '--height', help='Desired image height', required=False, default=img_height)
parser.add_argument('-s', '--save', help='Saves picture(s) to the specified directory', required=False, default=True)

args = parser.parse_args()
path = f"{args.dir}/{args.name}"
Path(path).mkdir(parents=True, exist_ok=True)
face_recognition(img_width = int(args.width), img_height= int(args.height), name = args.name, path=path, pictures=int(args.pictures), save_to_file=bool(args.save))
