import os
import core.animation as anim
from core.utils import match_length, inference_err, calculate_average_error
import numpy as np


def main():
    data_path = "./datasets/evaluation_dataset/mNIKI_dELMO/"
    result_path = "./datasets/evaluation_dataset/results/" + data_path
    os.makedirs(result_path, exist_ok=True)

    file_paths = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".bvh"):
                file_paths.append(os.path.join(root, file))

    gt_paths, output_paths = [], []

    for file_path in file_paths:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        if "Retargeted" in filename:
            output_paths.append(file_path)
        else:
            gt_paths.append(file_path)
    del file_paths

    gt_paths.sort()
    output_paths.sort()

    file_names, joint_names, lengths = [], [], []
    niki_pos_errs, niki_rot_errs, niki_linvel_errs, niki_angvel_errs = [], [], [], []

    for i in range(len(gt_paths)):
        gt, niki = anim.Animation(), anim.Animation()
        gt.load_bvh(gt_paths[i], ftrim=60, btrim=60, blender=True)
        niki.load_bvh(output_paths[i], ftrim=60, btrim=60, blender=True)

        match_length([gt, niki])

        file_names.append(os.path.splitext(os.path.basename(gt_paths[i]))[0])
        joint_names = np.insert(gt.joints, 0, 'length')

        gt.compute_world_transform(fix_root=True)
        niki.compute_world_transform(fix_root=True)

        niki_metrics = inference_err(niki, gt)

        lengths.append(niki_metrics[-1])

        niki_pos_errs.append(np.multiply(niki_metrics[8], niki_metrics[-1]))
        niki_rot_errs.append(np.multiply(niki_metrics[9], niki_metrics[-1]))
        niki_linvel_errs.append(np.multiply(niki_metrics[10], niki_metrics[-1]))
        niki_angvel_errs.append(np.multiply(niki_metrics[11], niki_metrics[-1]))

    niki_avg_pos_errs, niki_avg_rot_errs, niki_avg_linvel_errs, niki_avg_angvel_errs = \
    calculate_average_error(lengths, niki_pos_errs, niki_rot_errs, niki_linvel_errs, niki_angvel_errs, joint_names, file_names, result_path, "interp")

    print("----------NIKI----------")
    print("joint p: %f, joint r: %f, joint lv: %f, joint av: %f" % (np.mean(niki_avg_pos_errs[1:]), np.mean(niki_avg_rot_errs[1:]), np.mean(niki_avg_linvel_errs[1:]), np.mean(niki_avg_angvel_errs[1:])))
    # print("pelv p: %f, pelv r: %f, pelv lv: %f, pelv av: %f" % (niki_avg_pos_errs[0], niki_avg_rot_errs[0], niki_avg_linvel_errs[0], niki_avg_angvel_errs[0]))  # DO NOT USE in paper

if __name__ == "__main__":
    main()