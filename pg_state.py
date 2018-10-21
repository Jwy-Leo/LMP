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
from umbojson import Umbot7, get_dts_idx_list, get_iter_dts_loder
from umboimgs import Umbot7 as Umbot7_train

from feat_umnet import UmboUNet#
from  model import SegModel
import pdb

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=250, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--episode', type = int, default=5000, 
                    help="require a maximum number of playing the game")
parser.add_argument('--budget', type=int, default=16, help="requrie a budget for annotating")
parser.add_argument(
    '--n-class', type=int, default=2, metavar='',
    help='number of classes to segment (default: 2)')
parser.add_argument(
    '--base-planes', type=int, default=32, metavar='',
    help='number of feature planes for the first layer (default: 32)')
parser.add_argument(
    '--learning-rate', type=float, default=5e-5, metavar='',
    help='learning rate (default: 0.1)')
parser.add_argument(
    '--momentum', type=float, default=0.9, metavar='',
    help='SGD momentum (default: 0.9)')
parser.add_argument(
    '--decay-ratio', default=0.954999, type=float, metavar='',
    help='decay ratio for learning rate (default: 0.95499)')
parser.add_argument(
    '--pretrained-model-path', type=str, default='./Umbo_model_state_checkpoints.pth', metavar='',
    help='where to save the trained model')
parser.add_argument(
    '--save-model-dir', type=str, default='./saved_models', metavar='',
    help='where to save the trained model')
parser.add_argument(
    '--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument(
    '--data-step', type=int, default=0, help='for increasing the number of candidates, 0,1,2...')
parser.add_argument(
    '--use-cuda', action='store_true', help='use cuda?')
parser.add_argument(
    '--test', action='store_true', help='test mode?')
args = parser.parse_args()


#env = gym.make('CartPole-v0')
#env.seed(args.seed)
torch.manual_seed(args.seed)

class MEM_net(nn.Module):
    def write(self, input_feat, W_key, W_val):
        self.M_key = F.relu(W_key(input_feat))
        self.M_val = F.relu(W_val(input_feat))
    def read(self,num_obs, h_T):
        atten = torch.exp(torch.mm(self.M_key, h_T.transpose(0,1))) / torch.exp(torch.mm(self.M_key, h_T.transpose(0,1))).sum()
        output = (self.M_val * atten)
        return output

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
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(64, 16) 
        self.fc7 = nn.Linear(272, num_actions) 
        
        len_mem_emb= 64; #256; 
        self.W_key = nn.Linear(256, len_mem_emb)
        self.W_val = nn.Linear(256, len_mem_emb)
        self.Wh = nn.Linear(256, len_mem_emb)
        #self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x_im, x_prior,mem):
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
        self.feat = self.fc5(torch.cat([x,x2],1))
        
        self.feat_block = self.feat.detach()
        h_T = self.Wh(self.feat_block)
        memorize = mem.read(3, h_T).mean(0,keepdim=True)#.unsqueeze(0)
        act_obs = torch.cat([self.fc6(memorize),self.feat],1)
        return F.softmax(self.fc7(act_obs))


def finish_episode(terminal_reward):
    rewards = [terminal_reward] * len(saved_actions)
    rewards = Variable(torch.Tensor(rewards)).cuda()
    rewards = rewards *100#* pri
    try:
        print('weighted reward: %.4f')%(rewards.mean().data[0])
    except:
        print('weighted reward: 0.00')#%(rewards.mean().data[0])
        
    
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
if os.path.exists(args.pretrained_model_path):
    model_state_checkpoint = torch.load(args.pretrained_model_path)
    UNet.load_state_dict(model_state_checkpoint['best_model_state'])

if os.path.exists(
        os.path.join(args.saved_models,'policy_lr-4_gym_mem_model_state_checkpoint_xxxx.pth')):
    model_state_checkpoint_2 = torch.load(
        os.path.join(args.saved_models,'policy_lr-4_gym_mem_model_state_checkpoint_xxxx.pth'))

    policy.load_state_dict(model_state_checkpoint_2['last_model_state'])

# initialize optimzers
optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate) 


######
kwargs = {'num_workers': 1,
          'pin_memory': True} if torch.cuda.is_available() else {}
    
### data loader ###
data_root = '/media/addhd6/nitahaha/gym-20170820' 

#train data
# gradually incrasing the size of candidate pool
data_img_train = ['less_videos/rgb_frame_resized_crop256_clean_lvl1',
                 ,'less_videos/rgb_frame_resized_crop256_clean_lvl2',
                 ,'less_videos/rgb_frame_resized_crop256_clean_lvl3']

data_prior_train = 'less_videos/priors_resized_crop256_clean'

#val data
data_img = 'img_resized'
data_json_val = 'modified/rgb_resized_val_labels_crowd_rand100.json'

#test data
data_img_test = 'less_videos/ir_frame_resized_crop256_clean_more'
data_prior_test = 'less_videos/ir_priors_resized_crop256_clean_more'
#
data_idx = args.data_step
batch_size = 8
test_bz = 2

train_loader_list = []
train_loader_list.append(torch.utils.data.DataLoader(
        Umbot7_train(os.path.join(data_root, data_img_train[data_idx]),
               os.path.join(data_root, data_prior_train),
               args.n_class,train=True,resized=True),
        batch_size=1, shuffle=True, **kwargs))

#train on gym
val_loader_list = []
val_loader_list.append(torch.utils.data.DataLoader(
        Umbot7(os.path.join(data_root, data_img),
               os.path.join(data_root, data_json_val),
               args.n_class,train=False,resized=True),
        batch_size=test_bz, shuffle=False, **kwargs))
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

def TEST():#Pnet, Unet, test_loader):
 
    MEM = MEM_net().cuda()
    ## feed in target images/motion for testing
    test_loader_list = []
    test_loader_list.append(torch.utils.data.DataLoader(
            Umbot7_train(os.path.join(data_root, data_img_test),
                   os.path.join(data_root, data_prior_test),
                   args.n_class,train=True,resized=True),
            batch_size=1, shuffle=True, **kwargs))
    test_loader_iter, n_batch = get_train_loader_iter(test_loader_list) #get a new one if finished
    get = 0
    iters = 0
    test_saved_filename = []
    test_ignored_filename = []
    observ_buffer = []
    while  get < args.budget:
        try:
            img, target, file_name = next(test_loader_iter[0])
        ### convert to cuda and torch type
        except:
            # run out off data in loader
            break
        img, target = img.cuda(async=True), target.cuda(async=True)
        img, target = Variable(img), Variable(target)
        
        if not get:
            observ_buffer = Variable(torch.zeros(3,256).cuda())
            MEM.write(observ_buffer, policy.W_key, policy.W_val)
        
        probs = policy(img, target.unsqueeze(1).float(),MEM)

        
        action = torch.max(probs,1)[1]
        if action.data[0]:
            get += 1
            print('get')
            test_saved_filename.append(file_name)
            
            observ_buffer = torch.cat((observ_buffer,policy.feat),0)
            observ_buffer = observ_buffer[-3:,...]
        
        else : 
            print('skip')
            test_ignored_filename.append(file_name)
        ## save these ## 
        #saved_img_states.append(img)
        #saved_target_states.append(target.unsqueeze(1).float())
        #saved_actions.append(action)
        #saved_probs.append(probs)
    print('*** did not label ***', test_ignored_filename)
    print('*** label ***', test_saved_filename)
    return test_saved_filename

if args.test:
    print('test the policy on target domain')
    TEST()
    raise SystemExit

MEM = MEM_net().cuda()
#avg_score = 0.395 #0.3096
avg_score = 0.395 #0.3096
cal_avg_score = 0.
for i_episode in range(args.episode):
    ## initalize the memory list
    saved_img_states = []
    saved_target_states = []
    saved_actions = []
    saved_probs = []
    saved_filename = []
    observ_buffer = [] # save last 3 selected observ 
    train_loader_iter, n_batch = get_train_loader_iter(train_loader_list) #get a new one if finished
    get = 0
    while  get < args.budget:
        try:
            img, target, file_name = next(train_loader_iter[0])
        ### convert to cuda and torch type
        except:
            # run out off data in loader
            # cause out of mem ?
            break
        img, target = img.cuda(async=True), target.cuda(async=True)
        img, target = Variable(img), Variable(target)
        ### create last 3 observations
        if not get:
            observ_buffer = Variable(torch.zeros(3,256).cuda())
            MEM.write(observ_buffer, policy.W_key, policy.W_val)
        ###
        probs = policy(img, target.unsqueeze(1).float(),MEM)
        action = probs.multinomial()
        if action.data[0,0]:
            get += 1
            saved_filename.append(file_name)
            observ_buffer = torch.cat((observ_buffer,policy.feat),0)
            if observ_buffer.size(0) > 3:
                observ_buffer = observ_buffer[-3:,...]
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
    
    """
    if train_img_list != []:        
        train_imgs = torch.cat(train_img_list[:],0)
        train_targets = torch.cat(train_target_list[:],0)
        for iters in range(2):
            model.train(train_imgs, train_targets.long().squeeze(1))
    """
    if train_img_list != []:        
        if len(train_img_list) <= 8:    
            train_imgs = torch.cat(train_img_list[:],0)
            train_targets = torch.cat(train_target_list[:],0)
            for iters in range(3):
                model.train(train_imgs, train_targets.long().squeeze(1))
        else: 
            train_imgs_0 = torch.cat(train_img_list[:8],0)
            train_targets_0 = torch.cat(train_target_list[:8],0)
            train_imgs_1 = torch.cat(train_img_list[8:],0)
            train_targets_1 = torch.cat(train_target_list[8:],0)
            # training segmodel
            model.train(train_imgs_0, train_targets_0.long().squeeze(1))
            model.train(train_imgs_1, train_targets_1.long().squeeze(1))
            model.train(train_imgs_0, train_targets_0.long().squeeze(1))
            model.train(train_imgs_1, train_targets_1.long().squeeze(1))
    
    miou_p, miou_backg, avg_acc = model.target_test(val_loader_list)
    cal_avg_score += miou_p
    if not (i_episode+1) % 200:
        cal_avg_score /= 200.
        baseline_val = 0.395 #IOU accuracy of pretrained model on evaluation set
        avg_score = max(cal_avg_score, baseline_val) 
        cal_avg_score = 0.
        print('*****avg_score: ',avg_score)
    #print('test: person iou: %.4f')%(miou_p)
    # define reward
    reward = miou_p - avg_score 
    #reward = miou_p - 0.4461 #- 0.3096
    print('--- episode  %d\tget reward: %.4f')%(i_episode,reward)
    print(saved_filename)
    del saved_filename
    # train the policy net
    finish_episode(reward)
    # save policy net
    policy_model_state_checkpoint = {
            'episode': i_episode,
            'last_model_state': policy.state_dict()}
    if not ((i_episode+1)% args.log_interval) :    
        torch.save(policy_model_state_checkpoint,
            os.path.join(args.saved_models,
            'policy_lr-4_gym_mem_model_state_checkpoint_'+str(i_episode+1)+'.pth')) 
           
            

