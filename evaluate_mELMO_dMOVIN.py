import os
import core.animation as anim
from core.utils import match_length, inference_err, calculate_average_error
import numpy as np

def main():
    data_path = './datasets/evaluation_dataset/mELMO_dMOVIN/'
    result_path = './datasets/evaluation_dataset/results/' + data_path
    os.makedirs(result_path, exist_ok=True)

    file_paths = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".bvh"):
                file_paths.append(os.path.join(root, file))

    gt_paths, base_paths, future_paths, future_aug_paths = [], [], [], []

    for file_path in file_paths:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        if "model_baseline" in filename:
            base_paths.append(file_path)
        elif "model_latency" in filename:
            future_paths.append(file_path)
        elif "model_latsyn" in filename:
            future_aug_paths.append(file_path)
        else:
            gt_paths.append(file_path)
    del file_paths

    gt_paths.sort()
    base_paths.sort()
    future_paths.sort()
    future_aug_paths.sort()

    file_names, joint_names, lengths = [], [], []

    base_pos_errs, base_rot_errs, base_linvel_errs, base_angvel_errs = [], [], [], []
    future_pos_errs, future_rot_errs, future_linvel_errs, future_angvel_errs = [], [], [], []
    future_aug_pos_errs, future_aug_rot_errs, future_aug_linvel_errs, future_aug_angvel_errs = [], [], [], []
    
    for i in range(len(gt_paths)):
        gt, base, future, future_aug = anim.Animation(), anim.Animation(), anim.Animation(), anim.Animation()
        gt.load_bvh(gt_paths[i], ftrim=20, btrim=20)
        base.load_bvh(base_paths[i],upsample=-3, ftrim=60, btrim=60)
        future.load_bvh(future_paths[i], upsample=-3, ftrim=60, btrim=60)
        future_aug.load_bvh(future_aug_paths[i], upsample=-3, ftrim=60, btrim=60)

        match_length([gt, base, future, future_aug])

        file_names.append(os.path.splitext(os.path.basename(gt_paths[i]))[0])
        joint_names = np.insert(gt.joints, 0, 'length')

        gt.compute_world_transform(fix_root=True)
        base.compute_world_transform(fix_root=True)
        future.compute_world_transform(fix_root=True)
        future_aug.compute_world_transform(fix_root=True)

        base_metrics = inference_err(base, gt)
        future_metrics = inference_err(future, gt)
        future_aug_metrics = inference_err(future_aug, gt)

        lengths.append(base_metrics[-1])

        base_pos_errs.append(np.multiply(base_metrics[8], base_metrics[-1]))
        base_rot_errs.append(np.multiply(base_metrics[9], base_metrics[-1]))
        base_linvel_errs.append(np.multiply(base_metrics[10], base_metrics[-1]))
        base_angvel_errs.append(np.multiply(base_metrics[11], base_metrics[-1]))

        future_pos_errs.append(np.multiply(future_metrics[8], future_metrics[-1]))
        future_rot_errs.append(np.multiply(future_metrics[9], future_metrics[-1]))
        future_linvel_errs.append(np.multiply(future_metrics[10], future_metrics[-1]))
        future_angvel_errs.append(np.multiply(future_metrics[11], future_metrics[-1]))

        future_aug_pos_errs.append(np.multiply(future_aug_metrics[8], future_aug_metrics[-1]))
        future_aug_rot_errs.append(np.multiply(future_aug_metrics[9], future_aug_metrics[-1]))
        future_aug_linvel_errs.append(np.multiply(future_aug_metrics[10], future_aug_metrics[-1]))
        future_aug_angvel_errs.append(np.multiply(future_aug_metrics[11], future_aug_metrics[-1]))

    base_avg_pos_errs, base_avg_rot_errs, base_avg_linvel_errs, base_avg_angvel_errs = \
    calculate_average_error(lengths, base_pos_errs, base_rot_errs, base_linvel_errs, base_angvel_errs, joint_names, file_names, result_path, "base")
    future_avg_pos_errs, future_avg_rot_errs, future_avg_linvel_errs, future_avg_angvel_errs = \
    calculate_average_error(lengths, future_pos_errs, future_rot_errs, future_linvel_errs, future_angvel_errs, joint_names, file_names, result_path, "future")
    future_aug_avg_pos_errs, future_aug_avg_rot_errs, future_aug_avg_linvel_errs, future_aug_avg_angvel_errs = \
    calculate_average_error(lengths, future_aug_pos_errs, future_aug_rot_errs, future_aug_linvel_errs, future_aug_angvel_errs, joint_names, file_names, result_path, "future_aug")

    print("------------------ELMO Baseline------------------")
    print("joint p: %f, joint r: %f, joint lv: %f, joint av: %f" % (np.mean(base_avg_pos_errs[1:]), np.mean(base_avg_rot_errs[1:]), np.mean(base_avg_linvel_errs[1:]), np.mean(base_avg_angvel_errs[1:])))
    print("pelv p: %f, pelv r: %f, pelv lv: %f, pelv av: %f" % (base_avg_pos_errs[0], base_avg_rot_errs[0], base_avg_linvel_errs[0], base_avg_angvel_errs[0]))

    print("------------------ELMO Future------------------")
    print("joint p: %f, joint r: %f, joint lv: %f, joint av: %f" % (np.mean(future_avg_pos_errs[1:]), np.mean(future_avg_rot_errs[1:]), np.mean(future_avg_linvel_errs[1:]), np.mean(future_avg_angvel_errs[1:])))
    print("pelv p: %f, pelv r: %f, pelv lv: %f, pelv av: %f" % (future_avg_pos_errs[0], future_avg_rot_errs[0], future_avg_linvel_errs[0], future_avg_angvel_errs[0]))

    print("------------------ELMO Future Augmented------------------")
    print("joint p: %f, joint r: %f, joint lv: %f, joint av: %f" % (np.mean(future_aug_avg_pos_errs[1:]), np.mean(future_aug_avg_rot_errs[1:]), np.mean(future_aug_avg_linvel_errs[1:]), np.mean(future_aug_avg_angvel_errs[1:])))
    print("pelv p: %f, pelv r: %f, pelv lv: %f, pelv av: %f" % (future_aug_avg_pos_errs[0], future_aug_avg_rot_errs[0], future_aug_avg_linvel_errs[0], future_aug_avg_angvel_errs[0]))

if __name__ == "__main__":
    main()