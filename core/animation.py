import os
import re
import numpy as np
from scipy.spatial.transform import Rotation as R

def copy(self):
    cls = self.__class__
    result = cls.__new__(cls)
    result.__dict__.update(self.__dict__)
    return result

def compute_p_lerp(local_t_0, local_t_1, t):
    translation_0 = local_t_0[:3, 3]
    translation_1 = local_t_1[:3, 3]
    p_lerp = lerp(translation_0, translation_1, t)
    return p_lerp

def compute_r_slerp(local_t_0, local_t_1, t):
    rotation_0 = R.from_matrix(local_t_0[:3, :3]).as_quat()
    rotation_1 = R.from_matrix(local_t_1[:3, :3]).as_quat()
    r_slerp = slerp(rotation_0, rotation_1, t)
    r_slerp = R.from_quat(r_slerp)
    return r_slerp

def lerp(a, b, t):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    interpolated_vector = (1 - t) * a + t * b
    return interpolated_vector

def slerp(q1, q2, t):
    q1 = np.array(q1, dtype=np.float64)
    q2 = np.array(q2, dtype=np.float64)
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    dot_product = np.dot(q1, q2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    theta = np.arccos(dot_product)
    sin_theta = np.sin(theta)
    interpolated_quaternion = (
        np.sin((1 - t) * theta) / sin_theta * q1 +
        np.sin(t * theta) / sin_theta * q2
    )
    return interpolated_quaternion

def RPY2Quat(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qx, qy, qz, qw]

def XZProjection(mat):
    out = np.eye(4)
    basis_z = np.array([mat[0, 2], 0, mat[2, 2]])
    basis_y = np.array([[0, 1, 0]])
    basis_x = np.cross(basis_y, basis_z)
    pos = np.array([mat[0, 3], 0, mat[2, 3]])
    
    out[:3, 0] = basis_x / np.linalg.norm(basis_x)
    out[:3, 1] = basis_y / np.linalg.norm(basis_y)
    out[:3, 2] = basis_z / np.linalg.norm(basis_z)
    out[:3, 3] = pos
    return out

class Animation:
    def __init__(self):
        self.name = None
        self.coord = None
        self.fps = 0
        self.length = 0
        self.joints = []
        self.parents = []
        self.local_t = None
        self.world_t = None
        self.world_vw = None

    def load_bvh(self, path, euler = 'ZYX', upsample = 1, ftrim=0, btrim=0, blender=False):
        base = os.path.basename(path)
        self.name = os.path.splitext(base)[0]
        bvh = open(path, 'r')
        
        pose = []
        offsets = []
        current_joint = 0
        end_site = False
        for line in bvh:
            joint_line = re.match(r"ROOT\s+(\w+)", line)
            if joint_line == None:
                joint_line = re.match(r"\s*JOINT\s+(\w+)", line)

            if joint_line:
                self.joints.append(joint_line.group(1))
                self.parents.append(current_joint)
                current_joint = len(self.parents) - 1
                continue

            endsite_line = re.match(r"\s*End\sSite", line)
            if endsite_line:
                end_site = True
                continue

            offset_line = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
            if offset_line:
                if not end_site:
                    offsets.append(np.array([offset_line.group(1), offset_line.group(2), offset_line.group(3)]))
                continue

            if "}" in line:
                if end_site:
                    end_site = False
                else:
                    current_joint = self.parents[current_joint]
                continue

            if "Frames" in line:
                self.length = int(line.split(' ')[-1])
                continue

            if "Frame Time:" in line:
                self.fps = round(1 / float(line.split(' ')[-1]))
                continue

            if "HIERARCHY" in line or "{" in line or "CHANNELS" in line or "MOTION" in line:
                continue

            pose.append(np.array(line.strip().split(' ')))
        
        self.joints = np.asarray(self.joints, dtype=str)
        self.parents = np.asarray(self.parents, dtype=np.int8)
        
        pose = np.asarray(pose, dtype=np.float32)
        offsets = np.asarray(offsets, dtype=np.float32)
        
        # + 1 for root pos ori + shift
        if blender:
            pose = pose.reshape(self.length, self.joints.shape[0], 2, 3)
            root = pose[:, 0]
            joints = pose[:, 1:]
            joints_rot = joints[:, :, 1]
            pose = np.concatenate((root, joints_rot), axis=1)
        else:
            pose = pose.reshape(self.length, self.joints.shape[0] + 1, 3)

        self.length = pose.shape[0]
        self.local_t = np.zeros((self.length, self.joints.shape[0], 4, 4))
        for f in range(self.length):
            for j in range(1, self.joints.shape[0] + 1):
                local_t_mat = np.eye(4)
                r = R.from_euler(euler, pose[f, j], degrees=True)
                p = pose[f, 0] if j == 1 else offsets[j - 1]
                local_t_mat[:3, 3] = p
                local_t_mat[:3, :3] = r.as_matrix()
                self.local_t[f, j - 1] = local_t_mat
                
        # trim
        self.local_t = self.local_t[ftrim:-btrim] if btrim > 0 else self.local_t[ftrim:]
        self.length = self.local_t.shape[0]
        
        # upsample by lerp and slerp
        if upsample > 1:
            new_local_t = np.zeros(((self.length - 1) * upsample, self.joints.shape[0], 4, 4))
            for f in range(self.length - 1):
                for j in range(self.joints.shape[0]):
                    for k in range(upsample):
                        local_t_interp = np.eye(4)
                        local_t_0 = self.local_t[f, j]
                        local_t_1 = self.local_t[f + 1, j]
                        t = k / upsample
                        
                        p_lerp = compute_p_lerp(local_t_0, local_t_1, t)
                        r_slerp = compute_r_slerp(local_t_0, local_t_1, t)
                        
                        local_t_interp[:3, 3] = p_lerp
                        local_t_interp[:3, :3] = r_slerp.as_matrix()
                        new_local_t[f * upsample + k, j] = local_t_interp       
            self.local_t = new_local_t
        elif upsample < -1: # downsample
            self.local_t = self.local_t[::abs(upsample)]
        else:
            pass
        self.length = self.local_t.shape[0]

        print(f'Loaded {self.length} frames from {path}')

    def compute_world_transform(self, fix_root = True):
        self.world_t = np.zeros((self.length, self.joints.shape[0], 4, 4))
        for f in range(self.length):
            self.world_t[f, 0] = np.eye(4)
            
            start_idx = 0
            if fix_root:
                start_idx = 1
            
            for j in range(start_idx, self.joints.shape[0]):
                local_t = self.local_t[f, j]
                self.world_t[f, j] = np.matmul(self.world_t[f, self.parents[j]], local_t)
                
    def dup_upsample(self, n):
        # duplicate each frame n times
        dup_local_t = np.zeros((self.length * n, self.joints.shape[0], 4, 4))
        for f in range(self.length):
            for i in range(n):
                dup_local_t[f * n + i] = self.local_t[f]
        self.local_t = dup_local_t
        self.length = self.local_t.shape[0]