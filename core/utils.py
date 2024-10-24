import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as matanim
import matplotlib.colors as colors
import matplotlib.patheffects as pe
from scipy.spatial.transform import Rotation as R


def get_bvh_filepaths(datapath):
    filepaths = []
    for root, dirs, files in os.walk(datapath):
        for file in files:
            if file.endswith(".bvh"):
                filepaths.append(os.path.join(root, file))
    return filepaths

def save_to_csv(data, path, columns, index):
    df = pd.DataFrame(data, columns=columns, index=index)
    df.to_csv(path)

def match_length(list):
    min_length = min([x.length for x in list])
    for x in list:
        x.local_t = x.local_t[:min_length]
        x.length = min_length

def get_angle(v1, v2):
    v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)
    return np.arccos(np.clip(np.sum(v1 * v2, axis=-1), -1.0, 1.0)) * 180 / np.pi

def get_angle_mat(m1, m2):
    r = m2 @ np.linalg.inv(m1)
    r = R.from_matrix(r).as_rotvec()
    return np.linalg.norm(r, axis=-1) * 180 / np.pi

def inference_err(output, target):
    length = output.local_t.shape[0]
    
    # position - global
    pelv_pos = output.local_t[..., 0, :3, 3].reshape(length, 1, 3)
    gt_pelv_pos = target.local_t[..., 0, :3, 3].reshape(length, 1, 3)
    joint_pos = output.world_t[..., 1:, :3, 3]
    gt_joint_pos = target.world_t[..., 1:, :3, 3]
    pos = np.concatenate((pelv_pos, joint_pos), axis=1)    
    gt_pos = np.concatenate((gt_pelv_pos, gt_joint_pos), axis=1)
    pos_err = np.linalg.norm(pos - gt_pos, axis=-1)
    per_joint_pos_err = np.mean(pos_err, axis=0)
    
    avg_pelvis_pos_err = per_joint_pos_err[0]
    avg_joint_pos_err = np.mean(per_joint_pos_err[1:])
    
    # linear velocity - global
    pelv_linvel = pelv_pos[1:] - pelv_pos[:-1]
    gt_pelv_linvel = gt_pelv_pos[1:] - gt_pelv_pos[:-1]
    joint_linvel = joint_pos[1:] - joint_pos[:-1]
    gt_joint_linvel = gt_joint_pos[1:] - gt_joint_pos[:-1]
    linvel = np.concatenate((pelv_linvel, joint_linvel), axis=1)
    gt_linvel = np.concatenate((gt_pelv_linvel, gt_joint_linvel), axis=1)
    linvel_err = np.linalg.norm(linvel - gt_linvel, axis=-1)
    per_joint_linvel_err = np.mean(linvel_err, axis=0)
    avg_pelv_linvel_err = per_joint_linvel_err[0]
    avg_joint_linvel_err = np.mean(per_joint_linvel_err[1:])
    
    # rotation - local
    rot = output.local_t[..., :3, :3]
    gt_rot = target.local_t[..., :3, :3]
        
    gt_x_basis = gt_rot[..., :3, 0]
    gt_y_basis = gt_rot[..., :3, 1]
    x_basis = rot[..., :3, 0]
    y_basis = rot[..., :3, 1]
    
    x_err = get_angle(gt_x_basis, x_basis)
    y_err = get_angle(gt_y_basis, y_basis)
    rot_err = (x_err + y_err) / 2

    per_joint_rot_err = np.mean(rot_err, axis=0)
    avg_pelvis_rot_err = per_joint_rot_err[0]
    avg_joint_rot_err = np.mean(per_joint_rot_err[1:])
    
    # angular velocity - local
    angvel = rot[1:] @ np.linalg.inv(rot[:-1])
    gt_angvel = gt_rot[1:] @ np.linalg.inv(gt_rot[:-1])
    
    gt_angvel_x_basis = gt_angvel[..., :3, 0]
    gt_angvel_y_basis = gt_angvel[..., :3, 1]
    angvel_x_basis = angvel[..., :3, 0]
    angvel_y_basis = angvel[..., :3, 1]
    angvel_x_err = get_angle(gt_angvel_x_basis, angvel_x_basis)
    angvel_y_err = get_angle(gt_angvel_y_basis, angvel_y_basis)
    angvel_err = (angvel_x_err + angvel_y_err) / 2
    
    per_joint_angvel_err = np.mean(angvel_err, axis=0)
    avg_pelv_angvel_err = per_joint_angvel_err[0]
    avg_joint_angvel_err = np.mean(per_joint_angvel_err[1:])
    
    return avg_pelvis_pos_err, avg_pelvis_rot_err, avg_pelv_linvel_err, avg_pelv_angvel_err, \
           avg_joint_pos_err, avg_joint_rot_err, avg_joint_linvel_err, avg_joint_angvel_err, \
           per_joint_pos_err, per_joint_rot_err, per_joint_linvel_err, per_joint_angvel_err, length

def calculate_average_error(lengths, pos_errs, rot_errs, linvel_errs, angvel_errs, joint_names, file_names, result_path, prefix):
    """
    Calculate the average error and save it as a CSV.
    """
    lengths_arr = np.asarray(lengths)
    sum_lengths = np.sum(lengths_arr)
    
    pos_errs_arr = np.asarray(pos_errs)
    rot_errs_arr = np.asarray(rot_errs)
    linvel_errs_arr = np.asarray(linvel_errs)
    angvel_errs_arr = np.asarray(angvel_errs)
    
    avg_pos_errs = np.divide(np.sum(pos_errs_arr, axis=0), sum_lengths)
    avg_rot_errs = np.divide(np.sum(rot_errs_arr, axis=0), sum_lengths)
    avg_linvel_errs = np.divide(np.sum(linvel_errs_arr, axis=0), sum_lengths)
    avg_angvel_errs = np.divide(np.sum(angvel_errs_arr, axis=0), sum_lengths)
    
    pos_errs_arr = np.insert(pos_errs_arr, 0, lengths_arr, axis=1)
    rot_errs_arr = np.insert(rot_errs_arr, 0, lengths_arr, axis=1)
    linvel_errs_arr = np.insert(linvel_errs_arr, 0, lengths_arr, axis=1)
    angvel_errs_arr = np.insert(angvel_errs_arr, 0, lengths_arr, axis=1)
    
    # Create DataFrames and save as CSV
    pos_errs_df = pd.DataFrame(pos_errs_arr, index=file_names, columns=joint_names)
    rot_errs_df = pd.DataFrame(rot_errs_arr, index=file_names, columns=joint_names)
    linvel_errs_df = pd.DataFrame(linvel_errs_arr, index=file_names, columns=joint_names)
    angvel_errs_df = pd.DataFrame(angvel_errs_arr, index=file_names, columns=joint_names)
    
    pos_errs_df.to_csv(result_path + f'{prefix}_sum_per_joint_pos_err.csv')
    rot_errs_df.to_csv(result_path + f'{prefix}_sum_per_joint_rot_err.csv')
    linvel_errs_df.to_csv(result_path + f'{prefix}_sum_per_joint_linvel_err.csv')
    angvel_errs_df.to_csv(result_path + f'{prefix}_sum_per_joint_angvel_err.csv')

    return avg_pos_errs, avg_rot_errs, avg_linvel_errs, avg_angvel_errs

def animation_plot(motion, points, fps=20):
    motion.compute_world_transform(fix_root=False)
    parents = motion.parents

    Grot = motion.world_t[..., :3, :3]
    Gpos = motion.world_t[..., :3, 3]

    """ 60fps -> 20fps to synchronize with pcd data (20hz) for visualziation """
    Grot, Gpos = Grot[::3], Gpos[::3]
    scale = 30
    animation = Gpos * scale
            
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    rscale = scale * 1.5
    ax.set_xlim3d(-rscale, rscale)
    ax.set_zlim3d(0, rscale*2)
    ax.set_ylim3d(-rscale, rscale)

    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(20, -60) # (-40, 60): up view
    # ax.view_init(0, -90)
    ax.set_proj_type('ortho')

    # checkerboard pane
    facec = (254, 254, 254)
    linec = (240, 240, 240)
    facec = list(np.array(facec) / 256.0) + [1.0]
    linec = list(np.array(linec) / 256.0) + [1.0]

    ax.zaxis.set_pane_color(facec)
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    X = np.arange(-rscale, rscale, 10)
    Y = np.arange(-rscale, rscale, 10)
    xlen = len(X)
    ylen = len(Y)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(X.shape) # place it at a lower surface

    colortuple = (facec, linec)
    colors_pane = np.zeros((Z.shape + (4, )))
    for y in range(ylen):
        for x in range(xlen):
            colors_pane[y, x] = colortuple[(x + y) % len(colortuple)]

    ax.zaxis.line.set_lw(0.)
    ax.yaxis.line.set_lw(0.)
    ax.yaxis.line.set_color(linec)
    ax.xaxis.line.set_lw(0.)
    ax.xaxis.line.set_color(linec)

    acolors = list(sorted(colors.cnames.keys()))[::-1]
    acolors.pop(3)
    skel_lines = [plt.plot([0,0], [0,0], [0,0], color=acolors[0], zorder=3,
        lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()]
        )[0] for _ in range(animation.shape[1])]
    
    # Initialize the scatter plot
    if points is not None:
        dummy = np.zeros((1000, 3))
        scatter = ax.scatter(dummy[:,0], dummy[:,1], dummy[:,2], s=1.7, c='red', alpha=0.8, zorder=3)
    
    def animate(i):        
        changed = []
        for j in range(len(parents)):
            if parents[j] != -1:
                skel_lines[j].set_data(
                    [ animation[i,j,0],  animation[i,parents[j],0]],
                    [-animation[i,j,2], -animation[i,parents[j],2]])
                skel_lines[j].set_3d_properties(
                    [ animation[i,j,1],  animation[i,parents[j],1]])
        changed += skel_lines
        
        if points is not None:
            xyz = points[i][..., :3]
            # xyz = pad_or_sample_point_cloud(xyz, max_points=384, rgb=False)
            if len(xyz) < 1000:
                xyz = np.concatenate((xyz, np.zeros((1000-len(xyz), 3))))
            scatter._offsets3d = (xyz[:, 0]*scale, -xyz[:, 2]*scale, xyz[:, 1]*scale)

        # Set the title for the current timestep
        ax.set_title('Frame {}'.format(i+1), y=-0.01)

        return changed
        
    plt.tight_layout()
        
    ani = matanim.FuncAnimation(fig, animate, np.arange(len(animation)), interval=1000/fps)
        
    plt.show()
