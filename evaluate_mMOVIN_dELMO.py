import os
import core.animation as anim
from core.utils import match_length, inference_err, calculate_average_error
import numpy as np
import pandas as pd

def main():
    data_path = './datasets/evaluation_dataset/mMOVIN_dELMO/'
    result_path = './datasets/evaluation_dataset/results/' + data_path
    os.makedirs(result_path, exist_ok=True)

    file_paths = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".bvh"):
                file_paths.append(os.path.join(root, file))

    gt_paths, output_paths = [], []

    for file_path in file_paths:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        if "model_MOVIN" in filename:
            output_paths.append(file_path)
        else:
            gt_paths.append(file_path)    
    gt_paths.sort()
    output_paths.sort()

    file_names, joint_names, lengths = [], [], []

    interp_pos_errs, interp_rot_errs, interp_linvel_errs, interp_angvel_errs = [], [], [], []
    dup_pos_errs, dup_rot_errs, dup_linvel_errs, dup_angvel_errs = [], [], [], []

    for i in range(len(gt_paths)):
        gt, movin_interp, movin_dup = anim.Animation(), anim.Animation(), anim.Animation()
        gt.load_bvh(gt_paths[i], ftrim=60, btrim=60)
        movin_interp.load_bvh(output_paths[i], upsample=3, ftrim=20, btrim=20)
        movin_dup.load_bvh(output_paths[i], ftrim=20, btrim=20)
        movin_dup.dup_upsample(3)

        match_length([gt, movin_interp, movin_dup])

        file_names.append(os.path.splitext(os.path.basename(gt_paths[i]))[0])
        joint_names = np.insert(gt.joints, 0, 'length')

        gt.compute_world_transform(fix_root=True)
        movin_interp.compute_world_transform(fix_root=True)
        movin_dup.compute_world_transform(fix_root=True)

        interp_metrics = inference_err(movin_interp, gt)
        dup_metrics = inference_err(movin_dup, gt)

        lengths.append(interp_metrics[-1])

        interp_pos_errs.append(np.multiply(interp_metrics[8], interp_metrics[-1]))
        interp_rot_errs.append(np.multiply(interp_metrics[9], interp_metrics[-1]))
        interp_linvel_errs.append(np.multiply(interp_metrics[10], interp_metrics[-1]))
        interp_angvel_errs.append(np.multiply(interp_metrics[11], interp_metrics[-1]))

        dup_pos_errs.append(np.multiply(dup_metrics[8], dup_metrics[-1]))
        dup_rot_errs.append(np.multiply(dup_metrics[9], dup_metrics[-1]))
        dup_linvel_errs.append(np.multiply(dup_metrics[10], dup_metrics[-1]))
        dup_angvel_errs.append(np.multiply(dup_metrics[11], dup_metrics[-1]))

    interp_avg_pos_errs, interp_avg_rot_errs, interp_avg_linvel_errs, interp_avg_angvel_errs = \
    calculate_average_error(lengths, interp_pos_errs, interp_rot_errs, interp_linvel_errs, interp_angvel_errs, joint_names, file_names, result_path, "interp")
    dup_avg_pos_errs, dup_avg_rot_errs, dup_avg_linvel_errs, dup_avg_angvel_errs = \
    calculate_average_error(lengths, dup_pos_errs, dup_rot_errs, dup_linvel_errs, dup_angvel_errs, joint_names, file_names, result_path, "dup")

    print("----------MOVIN Duplicated----------")
    print("joint p: %f, joint r: %f, joint lv: %f, joint av: %f" % (np.mean(dup_avg_pos_errs[1:]), np.mean(dup_avg_rot_errs[1:]), np.mean(dup_avg_linvel_errs[1:]), np.mean(dup_avg_angvel_errs[1:])))
    print("pelv p: %f, pelv r: %f, pelv lv: %f, pelv av: %f" % (dup_avg_pos_errs[0], dup_avg_rot_errs[0], dup_avg_linvel_errs[0], dup_avg_angvel_errs[0]))
    
    print("----------MOVIN Interpolated----------")
    print("joint p: %f, joint r: %f, joint lv: %f, joint av: %f" % (np.mean(interp_avg_pos_errs[1:]), np.mean(interp_avg_rot_errs[1:]), np.mean(interp_avg_linvel_errs[1:]), np.mean(interp_avg_angvel_errs[1:])))
    print("pelv p: %f, pelv r: %f, pelv lv: %f, pelv av: %f" % (interp_avg_pos_errs[0], interp_avg_rot_errs[0], interp_avg_linvel_errs[0], interp_avg_angvel_errs[0]))


if __name__ == "__main__":
    main()