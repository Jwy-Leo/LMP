import argparse
#import gym
import sys
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
import copy
from datasets.umbojson import Umbot7, get_dts_idx_list, get_iter_dts_loder
from datasets.umboimgs import Umbot7 as Umbot7_train

from models.feat_umnet import UmboUNet#
from  models.model import SegModel
import pdb

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument(
    '--episode', type = int, default=10000, help="require a maximum number of playing the game")
parser.add_argument('--budget', type=int, default=60, help="requrie a budget for annotating")
parser.add_argument(
'--n-class', type=int, default=2, metavar='',
help='number of classes to segment (default: 2)')
parser.add_argument(
    '--base-planes', type=int, default=32, metavar='',
    help='number of feature planes for the first layer (default: 32)')
parser.add_argument(
    '--learning-rate', type=float, default=1e-5, metavar='',
    help='learning rate (default: 0.1)')
parser.add_argument(
    '--momentum', type=float, default=0.9, metavar='',
    help='SGD momentum (default: 0.9)')
parser.add_argument(
    '--decay-ratio', default=0.954999, type=float, metavar='',
    help='decay ratio for learning rate (default: 0.95499)')
parser.add_argument(
    '--pretrained-model-dir', type=str, default='../SegCow-dev/pytorch/output', metavar='',
    help='where to save the trained model')
parser.add_argument(
    '--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument(
    '--use-cuda', action='store_true', help='use cuda?')
parser.add_argument(
    '--test', action='store_true', help='test mode?')
parser.add_argument(
    '--test-img-dir', type=str, required=True, metavar='',
    help='where the testing images')
parser.add_argument(
    '--test-prior-dir', type=str, required=True, metavar='',
    help='where the testing priors')

args = parser.parse_args()


torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self, in_channels=3, num_actions=2):
        super(Policy, self).__init__()
        _m = 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32,64, kernel_size=5, stride=4, padding=1)
        self.conv3 = nn.Conv2d(64,128, kernel_size=3, stride=4, padding=1)
        
        self.conv1_im = nn.Conv2d(3, 32, kernel_size=5, stride=4, padding=1)
        self.conv2_im = nn.Conv2d(32,64, kernel_size=5, stride=4, padding=1)
        self.conv3_im = nn.Conv2d(64,128, kernel_size=3, stride=4, padding=1)
        self.w_im = nn.Linear(128*4*4, 32) # [ch*w*h, new_ch]
        self.fc4 = nn.Linear(128*4*4, 480) # [ch*w*h, new_ch]
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x_im, x_prior):
        x = x_prior
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x2 = x_im
        x2 = F.relu(self.conv1_im(x2))
        x2 = F.relu(self.conv2_im(x2))
        x2 = F.relu(self.conv3_im(x2))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x2 = F.relu(self.w_im(x2.view(x2.size(0), -1)))
        act_score = self.fc5(torch.cat([x,x2],1))
        return F.softmax(act_score)


def finish_episode(terminal_reward):
    rewards = [terminal_reward] * len(saved_actions)
    rewards = Variable(torch.Tensor(rewards)).cuda()
    rewards = rewards *100
    print('weighted reward: %.4f')%(rewards.mean().data[0])
    #rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps) # without minus baseline
    
    actions = []
    for act in saved_actions:
        if not act.data[0,0]:
            actions.append(torch.Tensor([1,0]))
        else: 
            actions.append(torch.Tensor([0,1]))
    actions = Variable(torch.stack(actions)).cuda()
    probs = torch.cat(saved_probs[:],0)
    rewards = rewards.expand(2,len(rewards)).transpose(0,1)
    
    surr_loss =  torch.mean(- torch.log(probs+1e-10)* actions * rewards)
    optimizer.zero_grad()
    surr_loss.backward()
    optimizer.step()
    
    print('surrogate loss: %.7f')%(surr_loss.data[0])
    del saved_img_states[:]
    del saved_target_states[:]
    del saved_actions[:]
    del saved_probs[:]

### set up model ###
args.use_cuda = True if args.use_cuda and torch.cuda.is_available() else False

cudnn.benchmark = True

if args.use_cuda:
    UNet = torch.nn.DataParallel(
        UmboUNet(n_class=args.n_class, base_planes=args.base_planes)).cuda()
    policy = Policy().cuda()
else:
    UNet = UmboUNet(n_class=args.n_class, base_planes=args.base_planes)
    policy = Policy()
# reload model
if os.path.exists(
        os.path.join(args.pretrained_model_dir, 'umbo_model_state_checkpoint.pth')):
    model_state_checkpoint = torch.load(
        os.path.join(args.pretrained_model_dir, 'umbo_model_state_checkpoint.pth'))

    UNet.load_state_dict(model_state_checkpoint['best_model_state'])
if os.path.exists(
        os.path.join('output', 'policy_lr-4_model_state_checkpoint_3650.pth')):
    model_state_checkpoint_2 = torch.load(
        os.path.join('output', 'policy_lr-4_model_state_checkpoint_3650.pth'))

    policy.load_state_dict(model_state_checkpoint_2['last_model_state'])
# initialize optimzers
optimizer = optim.Adam(policy.parameters(), lr=1e-4) 


######
kwargs = {'num_workers': 1,
          'pin_memory': True} if torch.cuda.is_available() else {}
    
### data loader ###
data_root = '/media/addhd6/nitahaha/gym-20170820' 
#train data
data_img_train = ['less_videos/rgb_frame_resized_crop256_clean_sub',
                    'images_crop256_clean',
                    'rgb_resized_train_img_crop256_clean_sub'] 
data_prior_train = ['less_videos/priors_resized_crop256_clean',
                    'UB_priors_crop256_clean',
                    'priors_resized_crop256_clean'] 
#val data
data_img = ['img_resized','images_test','rgb_resized_test_img_rand70']
data_json_val = ['modified/rgb_resized_val_labels_crowd_rand100.json','labels','rgb_resized_test_label']

#test data
data_img_test = ['less_videos/ir_frame_resized_crop256_clean','ir_resized_train_img_crop256_clean']
data_prior_test = ['less_videos/ir_priors_resized_crop256_clean','ir_priors_resized_crop256_clean']
#
data_idx = 0
batch_size = 8
test_bz = 2
train_loader_list = []
train_loader_list.append(torch.utils.data.DataLoader(
        Umbot7_train(os.path.join(data_root, data_img_train[data_idx]),
               os.path.join(data_root, data_prior_train[data_idx]),
               args.n_class,train=True,resized=True),
        batch_size=1, shuffle=True, **kwargs))

val_loader_list = []
val_loader_list.append(torch.utils.data.DataLoader(
        Umbot7(os.path.join(data_root, data_img[data_idx]),
               os.path.join(data_root, data_json_val[data_idx]),
               args.n_class,train=False,resized=True),
        batch_size=test_bz, shuffle=False, **kwargs))
"""
val_loader_list = []
val_loader_list.append(torch.utils.data.DataLoader(
        Umbot7_train(os.path.join(data_root_new, data_img[data_idx]),
               os.path.join(data_root_new, data_json_val[data_idx]),
               args.n_class,train=False,resized=True),
        batch_size=test_bz, shuffle=False, **kwargs))
"""
###################
# policy gradient #
###################
def get_train_loader_iter(train_loader_list):
    ### train loader list ###
    train_iter_dts_loader_list = get_iter_dts_loder(train_loader_list)
    train_dts_idx_list = get_dts_idx_list(train_loader_list)
    train_n_batch = len(train_dts_idx_list) 
    train_perm_sub_dts_idx = torch.randperm(train_n_batch)
    return train_iter_dts_loader_list, train_n_batch

def TEST():
 
## feed in target images/motion for testing
    test_loader_list = []
    test_loader_list.append(torch.utils.data.DataLoader(
            #Umbot7_train(os.path.join(data_root, data_img_test[0]),
            #       os.path.join(data_root, data_prior_test[0]),
            Umbot7_train(args.test_img_dir,
                   args.test_prior_dir,
                   args.n_class,train=True,resized=True),
            batch_size=1, shuffle=True, **kwargs))
    test_loader_iter, n_batch = get_train_loader_iter(test_loader_list) 
    get = 0
    test_saved_filename = []
    test_ignored_filename = []
    while  get < args.budget:
        try:
            img, target, file_name = next(test_loader_iter[0])
        except:
            # run out off data in loader
            break
        img, target = img.cuda(async=True), target.cuda(async=True)
        img, target = Variable(img), Variable(target)
        probs = policy(img, target.unsqueeze(1).float())
        
        action = torch.max(probs,1)[1]
        if action.data[0]:
            get += 1
            print('get')
            test_saved_filename.append(file_name)
        else : 
            print('skip')
            test_ignored_filename.append(file_name)
    print('*** did not label ***', test_ignored_filename)
    print('*** label ***', test_saved_filename)
    return test_saved_filename

if args.test:
    print('test the policy on target domain')
    TEST()
    raise SystemExit

for i_episode in range(args.episode):
    ## initalize the memory list
    saved_img_states = []
    saved_target_states = []
    saved_actions = []
    saved_probs = []
    saved_filename = []
   
    train_loader_iter, n_batch = get_train_loader_iter(train_loader_list) #get a new one if finished
    get = 0
    while  get < 8:
        try:
            img, target, file_name = next(train_loader_iter[0])
        except:
            # run out off data in loader
            break
        img, target = img.cuda(async=True), target.cuda(async=True)
        img, target = Variable(img), Variable(target)
        probs = policy(img, target.unsqueeze(1).float())
        action = probs.multinomial()
        if action.data[0,0]:
            get += 1
            saved_filename.append(file_name)
        ## save these ## 
        saved_img_states.append(img)
        saved_target_states.append(target.unsqueeze(1).float())
        saved_actions.append(action)
        saved_probs.append(probs)
    
    # end of iters
    # train the model
    model = SegModel(copy.deepcopy(UNet),'_none_','_none_')
    # pack the picked images to a batch
    train_img_list = []
    train_target_list = []
    
    for it in range(len(saved_actions)):
        if saved_actions[it].data[0,0]:
            train_img_list.append(saved_img_states[it])
            train_target_list.append(saved_target_states[it])
    if train_img_list != []:
        train_imgs = torch.cat(train_img_list[:],0)
        train_targets = torch.cat(train_target_list[:],0)
        
        for iters in range(2):
            model.train(train_imgs, train_targets.long().squeeze(1))
    
    miou_p, miou_backg, avg_acc = model.target_test(val_loader_list)
    reward = miou_p - - 0.3096
    print('--- episode  %d\tget reward: %.4f')%(i_episode+0,reward)
    print(saved_filename)
    del saved_filename
    
    finish_episode(reward)
    # save policy net
    policy_model_state_checkpoint = {
            'episode': i_episode,
            'last_model_state': policy.state_dict()}
    if not (i_episode % 100) :    
        torch.save(policy_model_state_checkpoint,
            os.path.join('save_models',
            'policy_lr-4_gym_model_state_checkpoint_'+str(i_episode+0)+'.pth')) #save policy
           
            

