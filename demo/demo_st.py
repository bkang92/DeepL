import sys
import os
import os.path as osp
import math
import argparse
import numpy as np
import cv2
import time
import yaml
import shutil

import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn


from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as m

# for Yolo v5
from PIL import Image

sys.path.insert(0, '..')
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, './st_net')
from config import cfg
from model import get_pose_net, get_root_net
from custom_dataset import generate_patch_image
from utils.pose_utils import process_bbox, pixel2cam
from utils.vis import vis_keypoints, vis_3d_multiple_skeleton
from utils.pose_utils import naive_pose_tracker, pose_mapping
from st_net.st_gcn import Model

from ms_utils import count_params, import_class, load_msg3d, actions

def x_rotation(vector,theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
    return np.dot(R,vector).transpose(1, 0)

def y_rotation(vector,theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R,vector).transpose(1, 0)

def z_rotation(vector,theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
    return np.dot(R,vector).transpose(1, 0)

def animate(skeleton):
    bones = tuple((i-1, j-1) for (i,j) in (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19)
    ))
    ax.clear()
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    for i, j in bones:
        joint_locs = skeleton[:,[i,j]]
        # plot them
        ax.plot(joint_locs[0],joint_locs[1],joint_locs[2], color='blue')

    # action_class = labels[1][index] + 1
    # action_name = actions[action_class]
    # plt.title('Skeleton {} Frame #{} of 300 from {}\n (Action {}: {})'.format(index, skeleton_index[0], args.dataset, action_class, action_name))
    # skeleton_index[0] += 1
    return ax

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0', dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, default='24', dest='test_epoch')
    parser.add_argument('--video', type=str, default='samples/1.mp4')
    parser.add_argument(
        '--config',
        default='./st_config/st_gcn/ntu-xview/test.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--st-model',
        default=None,
        help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')
    parser.add_argument(
        '--feeder',
        default='feeder.feeder',
        help='data loader will be used')

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(args).keys()
        for k in default_arg.keys():
            print(k)
            if k not in key:
                print('WRONG ARG: ', k)
                assert(k in key)
        parser.set_defaults(**default_arg)
    args = parser.parse_args()


    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'

    return args



# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True
temp = None
count = 0
pose_3d = [0]

# start(args)

# st_model = import_class(args.st_model)
st_model = Model(**args.model_args).cuda()

# shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
# shutil.copy2(os.path.join('.', __file__), self.arg.work_dir)

# st_model = st_model(**args.model_args).cuda()

with torch.no_grad():
    # msg3d.eval()
    st_model.eval()

if args.weights:
    st_model = load_msg3d(args, st_model)
    # st_model = st_model.load_state_dict(torch.load(args.weights))
# load_param_groups()

# MuCo joint set
joint_num = 21
joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )

# snapshot load
model_path = './snapshot_%d.pth.tar' % int(args.test_epoch)
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_pose_net(cfg, False, joint_num)
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'])
model.eval()

rootm_path = './snapshot_18.pth.tar' 
assert osp.exists(rootm_path), 'Cannot find model at ' + rootm_path
print('Load checkpoint from {}'.format(rootm_path))
rmodel = get_root_net(cfg, False)
rmodel = DataParallel(rmodel).cuda()
ckpt = torch.load(rootm_path)
rmodel.load_state_dict(ckpt['network'])
rmodel.eval()

yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).fuse().eval().cuda()
yolov5 = yolov5.autoshape()
pose_tracker = naive_pose_tracker()
# prepare input image
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])

vid_path = '' # or 0
cap = cv2.VideoCapture(args.video)
# cap = cv2.VideoCapture(0)


# normalized camera intrinsics
focal = [1500, 1500] # x-axis, y-axis
princpt = [cap.get(3)/2, cap.get(4)/2] # x-axis, y-axis
print('focal length: (' + str(focal[0]) + ', ' + str(focal[1]) + ')')
print('principal points: (' + str(princpt[0]) + ', ' + str(princpt[1]) + ')')
print(cap.get(4), cap.get(3))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi', fourcc, 30.0, (int(cap.get(3)),int(cap.get(4))))
act = None

# for each cropped and resized human image, forward it to PoseNet
output_pose_2d_list = []
output_pose_3d_list = []
frame_index = 0
start = time.time()

while True:
    output_pose_2d_list = []
    output_pose_3d_list = []
    bbox_list = []
    confs = []

    ok, original_img = cap.read()
    if ok:
        H, W, _ = original_img.shape
        for_det = original_img.copy()
        for_det = Image.fromarray(for_det.astype(np.uint8)) if isinstance(for_det, np.ndarray) else for_det
        
        with torch.no_grad():
            prediction = yolov5(for_det, size=640)

        for pred in prediction:
            if pred is not None:
                for *box, conf, c in pred:
                    if int(c) == 0:
                        bboxes = [float(x) for x in box]
                        bboxes = [bboxes[0], bboxes[1], bboxes[2] - bboxes[0], bboxes[3] - bboxes[1]]
                        bboxes = np.array(bboxes)
                        # print(conf.item())
                        if conf.item() > 0.5:
                            bbox_list.append(bboxes)
                            confs.append(conf.item())
        pose_3ds = np.zeros((len(bbox_list), 21, 3), dtype=np.float32)
        for n in range(len(bbox_list)):
            box = np.array(bbox_list[n], dtype=np.int32)
            cv2.rectangle(original_img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

            bbox = process_bbox(np.array(bbox_list[n]), original_img.shape[1], original_img.shape[0])
            img, img2bb_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, False) 
            img = transform(img).cuda()[None,:,:,:]
            k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*focal[0]*focal[1]/(bbox[2]*bbox[3]))]).astype(np.float32)
            k_value = torch.FloatTensor([k_value]).cuda()[None,:]

            # forward
            with torch.no_grad():
                pose_3d = model(img) # x,y: pixel, z: root-relative depth (mm)
                root_3d = rmodel(img, k_value)
            root_3d = root_3d[0].cpu().numpy()
            root_depth_list = root_3d[2]

            # inverse affine transform (restore the crop and resize)
            pose_3d = pose_3d[0].cpu().numpy()
            pose_3d_in = pose_3d[:, :2].copy()
            pose_3d[:,0] = pose_3d[:,0] / cfg.output_shape[1] * cfg.input_shape[1]
            pose_3d[:,1] = pose_3d[:,1] / cfg.output_shape[0] * cfg.input_shape[0]
            pose_3d_xy1 = np.concatenate((pose_3d[:,:2], np.ones_like(pose_3d[:,:1])),1)
            img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0,0,1]).reshape(1,3)))
            pose_3d[:,:2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            output_pose_2d_list.append(pose_3d[:,:2].copy())
            pose_3d_in = pose_3d[:, :2].copy()
            
            # root-relative discretized depth -> absolute continuous depth
            pose_3d[:,2] = (pose_3d[:,2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0]/2) + root_depth_list
            pose_3d = pixel2cam(pose_3d, focal, princpt)
            pose_z = pose_3d[:, 2]
               
            output_pose_3d_list.append(pose_3d.copy())
        # visualize 2d poses
            
        vis_img = original_img.copy()
        for n in range(len(bbox_list)):
            vis_kps = np.zeros((3,joint_num))
            vis_kps[0,:] = output_pose_2d_list[n][:,0]
            vis_kps[1,:] = output_pose_2d_list[n][:,1]
            vis_kps[2,:] = 1
            vis_img = vis_keypoints(vis_img, vis_kps, skeleton)
        
        if temp is not None and len(pose_3d) != 1:
            if abs(np.mean(pose_3d[:, 2]) - np.mean(temp[:, 2])) > 0 and temp.shape[0] != 1:
                pose_3d[:, 2] = temp[:, 2]
            temp = pose_3d
        vis_kps = np.array(output_pose_3d_list)
        if vis_kps is not None and len(pose_3d) != 1:
            vis_kps = pose_mapping(vis_kps)
            pose_3d = pose_mapping(np.expand_dims(pose_3d, axis=0))
            vis_kps[:, :, 0] = vis_kps[:, :, 0] / 2250 - 1.
            vis_kps[:, :, 2] = -vis_kps[:, :, 1] / 1050 - .35
            vis_kps[:, :, 1] = (pose_3d[:, :, 2] - 8500) / 1000# - .9

        frame_index += 1
        pose_tracker.update(vis_kps, frame_index, np.array(confs))
        data_numpy = pose_tracker.get_skeleton_sequence()

        if data_numpy is not None:
            data = torch.from_numpy(data_numpy.copy())
            data = data.unsqueeze(0)
            data = data.float().cuda().detach()  # (1, channel, frame, joint, person)

            output, feature = st_model.extract_feature(data)
            output = output[0]
            feature = feature[0]
            
            intensity = (feature*feature).sum(dim=0)**0.5
            intensity = intensity.cpu().detach().numpy()

            voting_label = output.sum(dim=3).sum(
            dim=2).sum(dim=1).argmax(dim=0)

            num_person = data.size(4)
            latest_frame_label = [output[:, :, :, m].sum(
                dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]

            duty = time.time() - start
            
            # print(voting_label.item()+1, voting_label.item()+1, latest_frame_label[0].item()+1)

            result = voting_label.item()+1

            act = actions[result]
            start = time.time()
            
            if act == 'falling' or act=='kicking something' or act=='staggering':
                cv2.putText(vis_img, 'Warning : {} detected'.format(act), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,  
                1, (0, 0, 255), 2, cv2.LINE_AA) 
                print(act, result)
            else:
                cv2.putText(vis_img, '{} detected'.format(act), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,  
                1, (255, 0, 0), 2, cv2.LINE_AA) 
            print(result)
            duty = time.time() - start
        cv2.imshow('output_pose', vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(vis_img)


    else:
        break
cap.release()
cv2.destroyAllWindows()
out.release()


