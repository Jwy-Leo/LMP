import os
import cv2
import sys
import numpy as np
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser(description='copy the selected priors to a new folder')
parser.add_argument(
    '--img_dir', type=str, required=True, metavar='', help='where images dir')
parser.add_argument(
    '--prior_dir', type=str, required=True, metavar='', help='where priors dir')
parser.add_argument(
    '--logfile', type=str, required=True, metavar='', help='give inference logfile')
parser.add_argument(
    '--name', type=str, required=True, metavar='', help='name for new folder')
args = parser.parse_args()

img_dir = args.img_dir #'../gym-20170820/less_videos/ir_frame_resized_crop256_clean/'
prior_dir = args.prior_dir #'../gym-20170820/less_videos/ir_priors_resized_crop256_clean/'

target = args.name

if not os.path.exists(target):
    os.makedirs(target)
if not os.path.exists(os.path.join(target,'imgs')):
    os.makedirs(os.path.join(target,'imgs'))
if not os.path.exists(os.path.join(target,'priors')):
    os.makedirs(os.path.join(target,'priors'))
if not os.path.exists(os.path.join(target,'label')):
    os.makedirs(os.path.join(target,'label'))
if not os.path.exists(os.path.join(target,'unlabel')):
    os.makedirs(os.path.join(target,'unlabel'))




f = open(args.logfile)
for line in f:
    if '*** did not label ***' in line:
        items_unlabel = [item.translate(None, '[ "\']') for item in line.strip(')\n').split(',')[1:] ]
    if '*** label ***' in line:
        items_label = [item.translate(None, '[ "\']') for item in line.strip(')\n').split(',')[1:] ]


for item in items_label:
    im = cv2.imread(os.path.join(img_dir,item))
    pri = cv2.imread(os.path.join(prior_dir,item))
    cat = np.concatenate([im,pri],1) 
    cv2.imwrite(os.path.join(target,'label',item), cat)
    copyfile(os.path.join(img_dir,item), os.path.join(target,'imgs',item))
    copyfile(os.path.join(prior_dir,item), os.path.join(target,'priors',item))

for item in items_unlabel:
    im = cv2.imread(os.path.join(img_dir,item))
    pri = cv2.imread(os.path.join(prior_dir,item))
    cat = np.concatenate([im,pri],1) 
    cv2.imwrite(os.path.join(target,'unlabel',item), cat)
 
