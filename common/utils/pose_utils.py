import torch
import numpy as np
from config import cfg
import copy

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def rigid_transform_3D(A, B):
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B)
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    t = -np.dot(R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return R, t

def rigid_align(A, B):
    R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(R, np.transpose(A))) + t
    return A2

def get_bbox(joint_img):
    # bbox extract from keypoint coordinates
    bbox = np.zeros((4))
    xmin = np.min(joint_img[:,0])
    ymin = np.min(joint_img[:,1])
    xmax = np.max(joint_img[:,0])
    ymax = np.max(joint_img[:,1])
    width = xmax - xmin - 1
    height = ymax - ymin - 1
    
    bbox[0] = (xmin + xmax)/2. - width/2*1.2
    bbox[1] = (ymin + ymax)/2. - height/2*1.2
    bbox[2] = width*1.2
    bbox[3] = height*1.2

    return bbox

def process_bbox(bbox, width, height):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = cfg.input_shape[1]/cfg.input_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    return bbox

def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]))

    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint


def fliplr_joints(_joints, width, matched_parts):
    """
    flip coords
    joints: numpy array, nJoints * dim, dim == 2 [x, y] or dim == 3  [x, y, z]
    width: image width
    matched_parts: list of pairs
    """
    joints = _joints.copy()
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()

    return joints

def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)


def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
                            dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    assert flipped.device == tensor.device
    assert flipped.requires_grad == tensor.requires_grad
    return flipped



# joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
#  1-base of the spine 2-middle of the spine
# 3-neck 4-head 5-left shoulder 6-left elbow 7-left wrist 8-
# left hand 9-right shoulder 10-right elbow 11-right wrist 12-
# right hand 13-left hip 14-left knee 15-left ankle 16-left foot 17-
# right hip 18-right knee 19-right ankle 20-right foot 21-spine 22-
# tip of the left hand 23-left thumb 24-tip of the right hand 25-
# right thumb

def pose_mapping(poses):
    # print(poses.shape)
    pose_ = np.zeros(poses.shape)
    # print(poses.shape)
    for n in range(poses.shape[0]):
        pose_[n, 0, :] = poses[n, 14, :]
        pose_[n, 1, :] = poses[n, 15, :]
        pose_[n, 2, :] = poses[n, 16, :]
        pose_[n, 3, :] = poses[n, 0, :]
        pose_[n, 4, :] = poses[n, 5, :]
        pose_[n, 5, :] = poses[n, 6, :]
        pose_[n, 6, :] = poses[n, 7, :]
        pose_[n, 7, :] = poses[n, 18, :]
        pose_[n, 8, :] = poses[n, 2, :]
        pose_[n, 9, :] = poses[n, 3, :]
        pose_[n, 10, :] = poses[n, 4, :]
        pose_[n, 11, :] = poses[n, 17, :]
        pose_[n, 12, :] = poses[n, 11, :]
        pose_[n, 13, :] = poses[n, 12, :]
        pose_[n, 14, :] = poses[n, 13, :]
        pose_[n, 15, :] = poses[n, 20, :]
        pose_[n, 16, :] = poses[n, 8, :]
        pose_[n, 17, :] = poses[n, 9, :]
        pose_[n, 18, :] = poses[n, 10, :]
        pose_[n, 19, :] = poses[n, 19, :]
        pose_[n, 20, :] = poses[n, 1, :]
        # pose_[22, :] = poses[16, :]
        # pose_[23, :] = poses[16, :]
        # pose_[24, :] = poses[16, :]
        # pose_[25, :] = poses[16, :]
    return pose_


class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    """

    def __init__(self, data_frame=128, num_joint=21, max_frame_dis=np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info = list()

    def update(self, multi_pose, current_frame, confs):
        # multi_pose.shape: (num_person, num_joint, 3)

        if current_frame <= self.latest_frame:
            return

        if len(multi_pose.shape) != 3:
            return

        # score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)
        score_order = confs.argsort(axis=0)
        for p in multi_pose[score_order]:

            # match existing traces
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                # trace.shape: (num_frame, num_joint, 3)
                if current_frame <= latest_frame:
                    continue
                mean_dis, is_close = self.get_dis(trace, p)
                if is_close:
                    if matching_trace is None:
                        matching_trace = trace_index
                        matching_dis = mean_dis
                    elif matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis

            # update trace information
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]

                # padding zero if the trace is fractured
                pad_mode = 'interp' if latest_frame == self.latest_frame else 'zero'
                pad = current_frame-latest_frame-1
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)

            else:
                new_trace = np.array([p])
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame

    def get_skeleton_sequence(self):

        # remove old traces
        valid_trace_index = []
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            if self.latest_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        self.trace_info = [self.trace_info[v] for v in valid_trace_index]

        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None

        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.latest_frame - latest_frame)
            d = trace[-end:]
            beg = end - len(d)
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))

        return data

    # concatenate pose to a trace
    def cat_pose(self, trace, pose, pad, pad_mode):
        # trace.shape: (num_frame, num_joint, 3)
        num_joint = pose.shape[0]
        num_channel = pose.shape[1]
        if pad != 0:
            if pad_mode == 'zero':
                trace = np.concatenate(
                    (trace, np.zeros((pad, num_joint, 3))), 0)
            elif pad_mode == 'interp':
                last_pose = trace[-1]
                coeff = [(p+1)/(pad+1) for p in range(pad)]
                interp_pose = [(1-c)*last_pose + c*pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    # calculate the distance between a existing trace and the input pose

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy)**2).sum(1))**0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close
