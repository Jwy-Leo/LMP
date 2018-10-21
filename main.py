from __future__ import print_function

import argparse
import copy
import os
import pdb
import click
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from models.adverser import critic
from models.feat_umnet import UmboUNet#
from datasets.umboimgs import Umbot7 as Umbot7_train
from datasets.umbojson import Umbot7, get_dts_idx_list, get_iter_dts_loder
from datasets.umbot7 import Umbot7 as Umbot7_src 
import numpy as np
import cv2
from torch.autograd import Function

parser = argparse.ArgumentParser(description='SegCow Training Routine')
parser.add_argument(
    '--n-class', type=int, default=2, metavar='',
    help='number of classes to segment (default: 2)')
parser.add_argument(
    '--base-planes', type=int, default=32, metavar='',
    help='number of feature planes for the first layer (default: 32)')
parser.add_argument(
    '--epochs', type=int, default=10, metavar='',
    help='number of epochs to train (default: 150)')
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
    '--seed', type=int, default=1, metavar='',
    help='random seed (default: 1)')
parser.add_argument(
    '--save-model-dir', type=str, default='adapt_logs', metavar='',
    help='where to save the trained model')
parser.add_argument(
    '--pretrained-model-dir', type=str, default='/media/addhd6/nitahaha/SegCow-dev/pytorch/output', metavar='',
    help='where to save the trained model')
parser.add_argument(
    '--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument(
    '--use-cuda', action='store_true', help='use cuda?')
parser.add_argument(
    '--visual', action='store_true', help='save visualization as inference?')
parser.add_argument(
    '--ca', action='store_true', help='add classwise alignment')
parser.add_argument(
    '--alpha', type=int, default=500, help='hyperp for GRL')
parser.add_argument(
    '--beta', type=float, default=0.1, help='weight for lr of feature extractor')
parser.add_argument(
    '--gamma', type=float, default=1, help='weight for lr of domain classifer')

args = parser.parse_args()
args.use_cuda = True if args.use_cuda and torch.cuda.is_available() else False

sup_dts_path = os.path.expanduser('../gym-20170820')
src_sup_dts_path = '/media/addhd4/nitahaha/coco/'

cudnn.benchmark = True

batch_size = 20
test_bz = 4

kwargs = {'num_workers': 1,
          'pin_memory': True} if torch.cuda.is_available() else {}
data_img = ['img_resized','img_resized']
data_json_train = ['modified/rgb_resized_train_labels.json','modified/ir_resized_train_labels.json']
data_json_val = ['modified/rgb_resized_val_labels_crowd.json','modified/ir_resized_val_labels_crowd.json']
data_idx = 1
###################################################
###               target loader                   ###
#####################################################
train_loader_list = []
train_loader_list.append(torch.utils.data.DataLoader(
        Umbot7(os.path.join(sup_dts_path, data_img[data_idx]),
               os.path.join(sup_dts_path, data_json_train[data_idx]),
               args.n_class,train=True,resized=True),
        batch_size=batch_size, shuffle=True, **kwargs))

val_loader_list = []
val_loader_list.append(torch.utils.data.DataLoader(
        Umbot7(os.path.join(sup_dts_path, data_img[data_idx]),
               os.path.join(sup_dts_path, data_json_val[data_idx]),
               args.n_class,train=False,resized=True),
        batch_size=test_bz, shuffle=False, **kwargs))


#####################################################
###               source loader                   ###
#####################################################
if args.base_planes == 32:
    # 2 GPUs
    sub_dts_list = {'128': 40, '192': 30, '256': 20, '384': 20, '512': 20}
    #sub_dts_list = {'128': 20, '192': 16, '256': 12, '384': 12, '512': 12}
elif args.base_planes == 48:
    # 2 GPUs
    sub_dts_list = {'128': 28, '192': 18, '256': 12, '384': 12, '512': 12}

train_src_loader_list = []
for sub_dts, batch_size in sub_dts_list.items():
    train_src_loader_list.append(torch.utils.data.DataLoader(
        Umbot7_src(os.path.join(src_sup_dts_path, 'img'),
               os.path.join(src_sup_dts_path, 'infos_train_%s.t7' % sub_dts),
               args.n_class),
        batch_size=batch_size, shuffle=True, **kwargs))
"""
val_src_loader_list = []
for sub_dts, batch_size in sub_dts_list.items():
    val_src_loader_list.append(torch.utils.data.DataLoader(
        Umbot7_src(os.path.join(src_sup_dts_path, 'img'),
               os.path.join(src_sup_dts_path, 'infos_val_%s.t7' % sub_dts),
               args.n_class),
        batch_size=batch_size, shuffle=True, **kwargs))
"""
#####################################################
###        another target supervison loader       ###
#####################################################
train_act_loader_list = []
train_act_loader_list.append(torch.utils.data.DataLoader(
        Umbot7_train_act(os.path.join('./filter_gym_ir/imgs'),
               os.path.join('./filter_gym_ir/priors'),
               args.n_class,train=True,resized=True),
        batch_size=4, shuffle=True, **kwargs))


if args.use_cuda:
    model = torch.nn.DataParallel(
        UmboUNet(n_class=args.n_class, base_planes=args.base_planes)).cuda()
    adv = torch.nn.DataParallel(critic()).cuda()
    adv_p = torch.nn.DataParallel(critic()).cuda()
    adv_b = torch.nn.DataParallel(critic()).cuda()

else:
    model = UmboUNet(n_class=args.n_class, base_planes=args.base_planes)
    adv = torch.nn.DataParallel(critic())
    adv_p = torch.nn.DataParallel(critic())
    adv_b = torch.nn.DataParallel(critic())


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
optim_act = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
optim_d = torch.optim.Adam([{'params':model.parameters(),'lr':args.learning_rate*args.beta},
                            {'params':adv.parameters()}], lr=args.learning_rate*args.gamma)
optim_d_p = torch.optim.Adam([{'params':model.parameters(),'lr':args.learning_rate*args.beta*args.ca},
                              {'params':adv_p.parameters()},{'params':adv_b.parameters()}],
                              lr=args.learning_rate*args.gamma)

if not os.path.exists(args.save_model_dir):
    os.mkdir(args.save_model_dir)
save_visual_dir = "./visual_GA"
if not os.path.exists(save_visual_dir):
    os.mkdir(save_visual_dir)


if os.path.exists(
        os.path.join(args.pretrained_model_dir, 'umbo_model_state_checkpoint.pth')):
    model_state_checkpoint = torch.load(
        os.path.join(args.pretrained_model_dir, 'umbo_model_state_checkpoint.pth'))

    epoch_start = model_state_checkpoint['epoch_end'] + 1

    train_log = model_state_checkpoint['train_log']
    val_log = model_state_checkpoint['val_log']

    model.load_state_dict(model_state_checkpoint['last_model_state'])
    if 'best_acc' in model_state_checkpoint:
        best_acc = model_state_checkpoint['best_acc']
    else:
        best_acc = float('-inf')

    if 'best_loss' in model_state_checkpoint:
        best_loss = model_state_checkpoint['best_loss']
    else:
        best_loss = float('inf')
else:
    epoch_start = 1
    train_log = {'acc': [], 'loss': []}
    val_log = {'acc': [], 'loss': []}
    best_acc = float('-inf')
    best_loss = float('inf')

def spatial_cross_entropy_criterion(input, target):
    n_class = input.data.size()[1]
    weights = torch.FloatTensor(n_class).fill_(1.0)
    weights[0] = 0.5
    if args.use_cuda:
        log_soft_max = nn.LogSoftmax().cuda()
        nll_loss_2d = nn.NLLLoss2d(weights).cuda()
    else:
        log_soft_max = nn.LogSoftmax()
        nll_loss_2d = nn.NLLLoss2d(weights)
    return nll_loss_2d(log_soft_max(input), target)

def save_visualize(predict, image, name, resized=False):
    class_id = predict.max(1)[1]
    class_id = class_id.cpu().data.numpy()
    image = image.cpu().data.numpy()
    image = image[0,...]
    class_id = class_id[0,...]
    class_id = np.stack([np.zeros_like(class_id),np.zeros_like(class_id),class_id],0)
    visual = 255.*(image*0.8 + class_id*0.5)
    cv2.imwrite(save_visual_dir +'/'+ name,np.transpose(visual,(1,2,0)))


def per_pixel_accuracy(output, target):
    _, predict = torch.max(output, 1)

    predict = predict.byte().squeeze()
    return float(torch.eq(predict, target.byte()).sum()) / float(
        target.nelement())

def iou_accuracy(output, target):
    _, predict = torch.max(output, 1)
    predict = predict.byte().squeeze()
    target = target.byte()
    iou_person = 1.
    iou_backg = 1.
    if float(torch.eq(target,1).sum()) != 0:
        p_person = (torch.eq(predict,1)*torch.eq(target,1)).sum()
        d_person = torch.eq(target,1).sum() + (torch.eq(predict,1)*torch.eq(target,0)).sum() 
        iou_person = float(p_person) / d_person
    if float(torch.eq(target,0).sum()) != 0:
        p_backg = (torch.eq(predict,0)*torch.eq(target,0)).sum() 
        d_backg = torch.eq(target,0).sum() + (torch.eq(predict,0)*torch.eq(target,1)).sum() 
        iou_backg = float(p_backg) / d_backg      
    return iou_person, iou_backg       

reverse_lamb = 0.0 #init value, will increase with step
#### implementation of gradient reversal operation ####
### for DA
class GradReverse(Function):
    def forward(self, x): 
        return x.view_as(x)
    def backward(self, grad_output):
        return (-reverse_lamb*grad_output)

def grad_reverse(x):
    return GradReverse()(x)

def train(epoch):
    global reverse_lamb
    model.train(mode=False)
    adv.train()
    adv_p.train()
    
    # get tgt data
    iter_dts_loader_list = get_iter_dts_loder(train_loader_list)
    dts_idx_list = get_dts_idx_list(train_loader_list)
    # get src data
    src_iter_dts_loader_list = get_iter_dts_loder(train_src_loader_list)
    src_dts_idx_list = get_dts_idx_list(train_src_loader_list)
    # get act data
    act_iter_dts_loader_list = get_iter_dts_loder(train_act_loader_list)
    act_dts_idx_list = get_dts_idx_list(train_act_loader_list)
 
    n_batch = min(len(dts_idx_list),len(src_dts_idx_list)) #choose smaller one as epoch size
    perm_sub_dts_idx = torch.randperm(n_batch)
    loss = Variable(torch.Tensor(1).zero_())
    domain_loss = Variable(torch.Tensor(1).zero_())
    acc = 0
    iou_person = 0
    iou_backg = 0
    d_acc = 0
    d_p_acc = 0
    d_b_acc = 0
    avg_loss = 0.0
    avg_d_loss = 0.0
    avg_acc = 0.0
    miou_person = 0.0
    miou_backg = 0.0

    def metrics_report_func(x):
        return 'acc=%.4f, loss=%.4f, d_loss=%.4f, d_acc=%.4f, dp_acc=%.4f, db_acc=%.4f' % (acc, loss.data[0], domain_loss.data[0],d_acc, d_p_acc , d_b_acc)#iou_person)#,iou_backg)

    with click.progressbar(iterable=perm_sub_dts_idx.tolist(), length=n_batch,
                           show_pos=False, show_percent=False, fill_char='#',
                           empty_char='-', label='train', width=0,
                           item_show_func=metrics_report_func) as bar:

        steps = 0 
        for i in bar:
            steps += 1
            
            img, target, filename = next(iter_dts_loader_list[dts_idx_list[i]])
            src_img, src_target = next(src_iter_dts_loader_list[src_dts_idx_list[i]])
             
            if args.use_cuda:
                img, target = img.cuda(async=True), target.cuda(async=True)
                src_img, src_target = src_img.cuda(async=True), src_target.cuda(async=True)
            
            #### model active learning part
            try:    
                act_img, act_target, act_file_name = next(act_iter_dts_loader_list[act_dts_idx_list[i]])
                act_img, act_target = act_img.cuda(async=True), act_target.cuda(async=True)
                act_img, act_target = Variable(act_img), Variable(act_target)
                act_feat, act_output = model(act_img)
                act_loss = 0.01*spatial_cross_entropy_criterion(act_output, act_target) #task loss
                optim_act.zero_grad()
                act_loss.backward(retain_variables=True)
                optim_act.step()
            except:
                pass
            
                               
            img, target = Variable(img), Variable(target)
            src_img, src_target = Variable(src_img), Variable(src_target)
            
            feat, output = model(img)
            src_feat, src_output = model(src_img)
            
            #predict = output.detach().max(1)[1]
            # calculate grid-level (pseudo) label 
            pool_fn = nn.AvgPool2d(16, stride=16,padding=0)
            mask = pool_fn(target.float())
            #mask = pool_fn(predict.float())
            src_mask = pool_fn(src_target.float())
            mask = mask.detach() 
            src_mask = src_mask.detach() 
            ##############################################
            #              adversarial trainning         #
            ##############################################
            # gradient reversal layer
            d_score_tgt = adv(grad_reverse(feat))
            d_score_src = adv(grad_reverse(src_feat))
            
            d_score_src_p = adv_p(grad_reverse(src_feat))
            d_score_src_b = adv_b(grad_reverse(src_feat))
            d_score_tgt_p = adv_p(grad_reverse(feat))
            d_score_tgt_b = adv_b(grad_reverse(feat))
            # domain labels
            d_label_src = torch.zeros(d_score_src.size()[0],d_score_src.size()[2],d_score_src.size()[3])
            d_label_tgt = torch.ones(d_score_tgt.size()[0],d_score_tgt.size()[2],d_score_tgt.size()[3])
            d_label_src = Variable(d_label_src.long()).cuda() 
            d_label_tgt = Variable(d_label_tgt.long()).cuda() 
            # calculate global domain loss
            d_loss_src = torch.nn.functional.cross_entropy(d_score_src,d_label_src)
            d_loss_tgt = torch.nn.functional.cross_entropy(d_score_tgt,d_label_tgt)

            domain_loss = (d_loss_src + d_loss_tgt)  
            loss = spatial_cross_entropy_criterion(src_output, src_target) #task loss
            
            # classwise 
            # assign classwise domain labels
            mask = mask.unsqueeze(dim=1) #probability_person of src
            src_mask = src_mask.unsqueeze(dim=1) #probability_person of tgt

            softlabel_src_p = torch.cat((src_mask, Variable(torch.zeros(src_mask.size()).cuda())), 1)
            softlabel_tgt_p = torch.cat((Variable(torch.zeros(mask.size()).cuda()), mask), 1)
            softlabel_src_b = torch.cat((1.-src_mask, Variable(torch.zeros(src_mask.size()).cuda())), 1)
            softlabel_tgt_b = torch.cat((Variable(torch.zeros(mask.size()).cuda()), 1.-mask), 1)
            # calculate classwise domain loss
            log_softmax = torch.nn.LogSoftmax()
            d_score_src_p = log_softmax(d_score_src_p)
            d_score_src_b = log_softmax(d_score_src_b)
            d_score_tgt_p = log_softmax(d_score_tgt_p)
            d_score_tgt_b = log_softmax(d_score_tgt_b)
            
            d_loss_src_p = torch.sum(- softlabel_src_p * d_score_src_p ) / src_mask.sum() 
            d_loss_src_b = torch.sum(- softlabel_src_b * d_score_src_b) / (1.-src_mask).sum()
            d_loss_tgt_p = torch.sum(- softlabel_tgt_p * d_score_tgt_p) / mask.sum()
            d_loss_tgt_b = torch.sum(- softlabel_tgt_b * d_score_tgt_b) / (1.-mask).sum()
            
            domain_loss_p = (d_loss_src_p + d_loss_tgt_p) 
            domain_loss_b = (d_loss_src_b + d_loss_tgt_b)

            #update Discriminator
            optim_d.zero_grad()
            (domain_loss).backward(retain_variables=True)
            optim_d.step()
            
            optim_d_p.zero_grad()
            (0.1*(domain_loss_p+domain_loss_b)).backward(retain_variables=True)
            optim_d_p.step()
            #calculate gradient reversal weight
            lamb = 2. / (1. + np.exp(-10.*((epoch)*n_batch+steps)/(n_batch*args.epochs*args.alpha))) -1
            reverse_lamb = lamb 
            #update model params by task loss
            optim.zero_grad()
            loss.backward(retain_variables=True)
            optim.step()
            
            acc = per_pixel_accuracy(output.data, target.data)
            iou_person, iou_backg = iou_accuracy(output.data, target.data)
            #### domain accuracy ####
            d_acc =  ((1-d_score_src.max(1)[1].float()).mean()+d_score_tgt.max(1)[1].float().mean()).data[0] / 2.
            d_p_acc =  ((1-d_score_src_p.max(1)[1].float()).mean()+d_score_tgt_p.max(1)[1].float().mean()).data[0] / 2.
            d_b_acc =  ((1-d_score_src_b.max(1)[1].float()).mean()+d_score_tgt_b.max(1)[1].float().mean()).data[0] / 2.
            
            avg_loss += loss.data[0]
            avg_d_loss += domain_loss.data[0]
            avg_acc += acc
            miou_person += iou_person
            miou_backg += iou_backg
    # Adaptive learning rate and weight decay
    for param_group in optim.param_groups:
        param_group['lr'] = param_group['lr'] * args.decay_ratio
    for param_group in optim_d.param_groups:
        param_group['lr'] = param_group['lr'] * args.decay_ratio
    for param_group in optim_d_p.param_groups:
        param_group['lr'] = param_group['lr'] * args.decay_ratio

    avg_acc /= n_batch
    avg_loss /= n_batch
    avg_d_loss /= n_batch
    miou_person /=n_batch
    miou_backg /= n_batch
    print('avg. per-pixel accuracy = %.4f' % avg_acc)
    print('avg. iou person = %.4f\t iou backg = %.4f' %(miou_person,miou_backg))
    print('avg. loss = %.4f  avg. domain loss = %.4f\n' %(avg_loss,avg_d_loss))

    train_log['acc'].append(avg_acc)
    train_log['loss'].append(avg_loss)


def val():

    model.eval()

    iter_dts_loader_list = get_iter_dts_loder(val_loader_list)
    dts_idx_list = get_dts_idx_list(val_loader_list)
    n_batch = len(dts_idx_list)
    perm_sub_dts_idx = torch.randperm(n_batch)

    loss = Variable(torch.Tensor(1).zero_())
    acc = 0
    iou_person = 0.0
    iou_backg = 0.0

    avg_loss = 0.0
    avg_acc = 0.0
    miou_person = 0.0
    miou_backg = 0.0
    def metrics_report_func(x):
        return 'acc=%.4f, miou_person = %.4f, miou_back = %.4f, loss=%.4f' % (acc,iou_person,iou_backg, loss.data[0])

    with click.progressbar(iterable=perm_sub_dts_idx.tolist(), length=n_batch,
                           show_pos=True, show_percent=True, fill_char='#',
                           empty_char='-', label='val', width=0,
                           item_show_func=metrics_report_func) as bar:

        for i in bar:
            idx = dts_idx_list[i]
            img, target, file_name = next(iter_dts_loader_list[idx])
            
            if args.use_cuda:
                img, target = img.cuda(async=True), target.cuda(async=True)

            img, target = Variable(img, volatile=True), Variable(target,
                                                                 volatile=True)
            feat , output = model(img)
            
            loss = spatial_cross_entropy_criterion(output, target)
            acc = per_pixel_accuracy(output.data, target.data)
            iou_person, iou_backg  = iou_accuracy(output.data, target.data)
            
            avg_loss += loss.data[0]
            avg_acc += acc
            
            miou_backg += iou_backg
            miou_person += iou_person
            #### visualization ####
            if args.visual:
                save_visualize(output, img, file_name[0].split('/')[-1],resized=True)
    avg_acc /= n_batch
    avg_loss /= n_batch
    miou_person /= n_batch
    miou_backg /= n_batch

    print('avg. per-pixel accuracy = %.4f' % avg_acc)
    print('avg. mIOU person, background = %.4f, %.4f' %(miou_person, miou_backg))
    print('avg. loss = %.4f' % avg_loss)

    val_log['acc'].append(avg_acc)
    val_log['loss'].append(avg_loss)

    return miou_person, avg_acc, avg_loss

best_iou = 0.
val()
for epoch in range(15): #args.epochs):#epoch_start, args.epochs+1):

    print('Epoch %d' % (epoch+1))
    print ('evaluation')
    train(epoch)
    iou, acc, loss = val()

    model_state_checkpoint = {
        'epoch_end': epoch,
        'train_log': train_log,
        'val_log': val_log,
        'last_model_state': model.state_dict(),
    }

    print()
    
    torch.save(model_state_checkpoint,
               os.path.join(args.save_model_dir,
                            'classwise_model_state_checkpoint.pth'))
    if iou > best_iou:
        best_iou = iou
        torch.save(model_state_checkpoint,
               os.path.join(args.save_model_dir,
                            'best_classwise_model_state_checkpoint.pth'))
