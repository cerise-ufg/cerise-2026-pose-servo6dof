import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
import cv2
import numpy as np
import torch
from lib.config import LOCAL_DATA_DIR
from lib.dataset.const import (INITIAL_JOINT_ANGLE, INTRINSICS_DICT,
                               JOINT_NAMES, JOINT_TO_KP)
from lib.dataset.dream import DreamDataset
from torch.utils.data import DataLoader
from lib.models.ctrnet.mask_inference import seg_mask_inference
from lib.models.full_net import get_rootNetwithRegInt_model
from lib.utils.BPnP import BPnP_m3d
from lib.utils.transforms import point_projection_from_3d_tensor
from lib.utils.urdf_robot import URDFRobot
from lib.utils.utils import cast, set_random_seed, create_logger
from tqdm import tqdm
from lib.utils.metrics import compute_metrics_batch, summary_add_pck

from lib.utils.geometries import (angle_axis_to_rotation_matrix,
                                  compute_geodesic_distance_from_two_matrices,
                                  rot6d_to_rotmat, rotmat_to_rot6d)
from pathlib import Path
# Variáveis globais para best/worst
best_accuracy = -1
worst_accuracy = 2
best_render = None
worst_render = None
best_name = ""
worst_name = ""
best_idx = -1

def make_mask(args):
    global best_accuracy, worst_accuracy, best_render, worst_render, best_name, worst_name, best_idx,worst_idx

    torch.autograd.set_detect_anomaly(True)
    set_random_seed(808)

    save_folder, ckpt_folder, log_folder, writer = create_logger(args)
    urdf_robot_name = args.urdf_robot_name
    robot = URDFRobot(urdf_robot_name)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    test_ds_name_real = args.train_ds_names
    rootnet_hw = (int(args.rootnet_image_size), int(args.rootnet_image_size))
    other_hw = (int(args.other_image_size), int(args.other_image_size))

    ds_test_real = DreamDataset(test_ds_name_real, rootnet_resize_hw=rootnet_hw, other_resize_hw=other_hw,
                                color_jitter=False, rgb_augmentation=False, occlusion_augmentation=False,
                                process_truncation=args.fix_truncation)
    ds_iter_test_real = DataLoader(ds_test_real, batch_size=args.batch_size, num_workers=args.n_dataloader_workers)

    model = get_rootNetwithRegInt_model({
        "robot_type": urdf_robot_name,
        "pose_params": INITIAL_JOINT_ANGLE,
        "cam_params": np.eye(4, dtype=float),
        "init_pose_from_mean": True
    }, args)

    seg_net = seg_mask_inference(INTRINSICS_DICT["realsense"], "realsense").to(device)

    checkpoint = torch.load(f"{args.save_path}/best_model_auc.pk")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.requires_grad = False

    vis_folder = os.path.join(args.save_path, 'vis')
    train_vispath = os.path.join(vis_folder, f"train")
    os.makedirs(train_vispath, exist_ok=True)

    def farward_loss(args, input_batch, device, model, seg_net, robot):
        model.eval()
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                module.eval()

        dtype = torch.float
        root_images = cast(input_batch["root"]["images"], device).float() / 255.
        root_K = cast(input_batch["root"]["K"], device).float()

        reg_images = cast(input_batch["other"]["images"], device).float() / 255.
        other_K = cast(input_batch["other"]["K"], device).float()

        images_id = input_batch["image_id"]
        images_original = cast(input_batch["images_original"], device).float() / 255.
        images_original_255 = images_original * 255.
        TCO = cast(input_batch["TCO"], device).float()
        K_original = cast(input_batch["K_original"], device).float()
        gt_jointpose = input_batch["jointpose"]
        gt_keypoints2d_original = cast(input_batch["keypoints_2d_original"], device).float()
        valid_mask = cast(input_batch["valid_mask"], device).float()
        gt_keypoints2d = cast(input_batch["other"]["keypoints_2d"], device).float()
        valid_mask_crop = cast(input_batch["other"]["valid_mask_crop"], device).float()
        gt_keypoints3d = cast(input_batch["other"]["keypoints_3d"], device).float()
        depth_feats = cast(input_batch['depth_feats'], device).float() / 255
        batch_size = root_images.shape[0]
        robot_type = "panda"
        robot = URDFRobot(robot_type)
        bboxes = cast(input_batch["root"]["bbox_gt2d_extended"], device).float()
        gt_pose = []
        gt_rot = []
        gt_trans = []
        for n in range(batch_size):
            jointpose = torch.as_tensor([gt_jointpose[k][n] for k in JOINT_NAMES[robot_type]])
            jointpose = cast(jointpose, device, dtype=dtype)
            rot6D = rotmat_to_rot6d(TCO[n, :3, :3])
            trans = TCO[n, :3, 3]
            gt_pose.append(jointpose)
            gt_rot.append(rot6D)
            gt_trans.append(trans)
        gt_pose = torch.stack(gt_pose, 0).to(torch.float32)
        gt_rot = torch.stack(gt_rot, 0).to(torch.float32)
        gt_trans = torch.stack(gt_trans, 0).to(torch.float32)

        world_3d_pts = robot.get_keypoints_only_fk(gt_pose)
        out = BPnP_m3d.apply(gt_keypoints2d_original, world_3d_pts, K_original[0])
        out_rot = angle_axis_to_rotation_matrix(out[:, 0:3])[:, :3, :3]
        out_rot = rotmat_to_rot6d(out_rot)
        # print("out_rot - gt_rot:", torch.sum(torch.abs(out_rot - gt_rot)))
        gt_rot = cast(out_rot, device).float()
        reference_keypoint_id = 3
        assert (reference_keypoint_id < len(robot.link_names)), reference_keypoint_id
        gt_root_trans = gt_keypoints3d[:, reference_keypoint_id, :]
        gt_root_rot = robot.get_rotation_at_specific_root(gt_pose, gt_rot, gt_trans, root=reference_keypoint_id)
        assert (gt_root_trans.shape == (batch_size, 3)), gt_root_trans
        gt_root_depth = gt_root_trans[:, 2].unsqueeze(-1)
        gt_root_uv = gt_keypoints2d[:, reference_keypoint_id, 0:2]

        real_bbox = torch.tensor([1000.0, 1000.0]).to(torch.float32)
        fx, fy = root_K[:, 0, 0], root_K[:, 1, 1]
        assert (fx.shape == fy.shape and fx.shape == (batch_size,)), (fx.shape, fy.shape)
        area = torch.max(torch.abs(bboxes[:, 2] - bboxes[:, 0]), torch.abs(bboxes[:, 3] - bboxes[:, 1])) ** 2
        assert (area.shape == (batch_size,)), area.shape
        k_values = torch.tensor(
            [torch.sqrt(fx[n] * fy[n] * real_bbox[0] * real_bbox[1] / area[n]) for n in range(batch_size)]).to(
            torch.float32)

        model.to(device)
        model.float()
        device_id = [0]
        model = torch.nn.DataParallel(model, device_ids=device_id, output_device=device_id[0])

        joint_to_kp_indice = JOINT_TO_KP[robot_type]
        assert len(joint_to_kp_indice) == robot.dof, (len(joint_to_kp_indice), robot.dof)
        joint_valid_mask = valid_mask[:, joint_to_kp_indice]
        gt_pose_before_mask = gt_pose.clone()

        pred_pose, pred_rot, pred_trans, pred_root_uv, pred_root_depth, \
            pred_uvd, pred_keypoints3d_int, pred_keypoints3d = model(reg_images, root_images, depth_feats, k_values,
                                                                     K=other_K)
        pred_keypoints2d_reproj_int = point_projection_from_3d_tensor(other_K, pred_keypoints3d_int)
        pred_keypoints2d_reproj = point_projection_from_3d_tensor(other_K, pred_keypoints3d)

        assert (pred_pose.shape == gt_pose.shape), f"{pred_pose.shape},{gt_pose.shape}"
        assert (pred_rot.shape == gt_root_rot.shape), f"{pred_rot.shape},{gt_root_rot.shape}"
        assert (pred_trans.shape == gt_root_trans.shape), f"{pred_trans.shape},{gt_root_trans.shape}"
        assert (pred_root_uv.shape == gt_root_uv.shape), f"{pred_root_uv.shape},{gt_root_uv.shape}"
        assert (pred_root_depth.shape == gt_root_depth.shape), f"{pred_root_depth.shape},{gt_root_depth.shape}"
        assert (pred_keypoints3d.shape == gt_keypoints3d.shape), f"{pred_keypoints3d.shape},{gt_keypoints3d.shape}"
        assert (
                pred_keypoints2d_reproj.shape == gt_keypoints2d.shape), f"{pred_keypoints2d_reproj.shape},{gt_keypoints2d.shape}"

        # metrics during validation
        image_dis3d_avg, image_dis2d_avg, batch_dis3d_avg, batch_dis2d_avg, \
            batch_l1jointerror_avg, image_l1jointerror_avg, root_depth_error, batch_error_relative, _ = compute_metrics_batch(
            robot=robot,
            gt_keypoints3d=gt_keypoints3d,
            gt_keypoints2d=gt_keypoints2d_original,
            K_original=K_original,
            gt_joint=gt_pose_before_mask,
            pred_joint=pred_pose,
            pred_rot=pred_rot,
            pred_trans=pred_trans,
            pred_depth=None,
            pred_xy=None,
            pred_xyz_integral=None,
            reference_keypoint_id=reference_keypoint_id
        )
        image_dis3d_avg_int, image_dis2d_avg_int = np.array([0], dtype=np.float), np.array([0], dtype=np.float)
        if pred_keypoints3d_int is not None:
            image_dis3d_avg_int, image_dis2d_avg_int, batch_dis3d_avg_int, batch_dis2d_avg_int, \
                batch_l1jointerror_avg_int, image_l1jointerror_avg_int, root_depth_error_int, batch_error_relative_int, _ = compute_metrics_batch(
                robot=robot,
                gt_keypoints3d=gt_keypoints3d,
                gt_keypoints2d=gt_keypoints2d_original,
                K_original=K_original,
                gt_joint=gt_pose_before_mask,
                pred_joint=None,
                pred_rot=None,
                pred_trans=None,
                pred_depth=None,
                pred_xy=None,
                pred_xyz_integral=pred_keypoints3d_int,
                reference_keypoint_id=reference_keypoint_id
            )
        rotation_diff = torch.mean(
            compute_geodesic_distance_from_two_matrices(rot6d_to_rotmat(pred_rot), rot6d_to_rotmat(gt_rot)))
        

        # máscaras
        seg_masks = seg_net(images_original_255).detach()
        gpu_renderer = robot.set_robot_renderer(K_original[0], device="cuda")
        robot_mesh_batch = robot.get_robot_mesh_list(joint_angles=pred_pose, renderer=gpu_renderer)
        rendered_masks = []
        for i in range(batch_size):
            rendered_mask = robot.get_rendered_mask_single_image_at_specific_root(
                pred_pose[i], pred_rot[i], pred_trans[i], robot_mesh_batch[i], gpu_renderer, root=3
            )
            rendered_masks.append(rendered_mask)

        metric_dict = {
            "image_dis2d_avg": image_dis2d_avg,
            "image_dis3d_avg": image_dis3d_avg,
        }

        return pred_keypoints2d_reproj, reg_images, seg_masks, rendered_masks, metric_dict

    # --------------------------
    # Loop principal
    # --------------------------
    global_best_error = float("inf")
    global_worst_error = -float("inf")
    global_best = {}
    global_worst = {}

    robot = URDFRobot("panda")
    seg_net = seg_mask_inference(INTRINSICS_DICT["realsense"], "realsense")

    for batchid, sample in enumerate(tqdm(ds_iter_test_real)):
        pred_keypoints2d_reproj, reg_images, seg_masks, rendered_masks, metric_dict = farward_loss(
            args=args, input_batch=sample, device=device, model=model, seg_net=seg_net, robot=robot
        )

        errors = metric_dict["image_dis2d_avg"]
        errors = errors if isinstance(errors, np.ndarray) else errors.detach().cpu().numpy()

        best_idx = np.argmin(errors)
        worst_idx = np.argmax(errors)

        # atualizar best
        if errors[best_idx] < global_best_error:
            global_best_error = errors[best_idx]
            global_best = {
                "img": (np.transpose(reg_images[best_idx].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8),
                "kps": pred_keypoints2d_reproj[best_idx].detach().cpu().numpy(),
                "seg": seg_masks[best_idx].detach().cpu().numpy(),
                "render": rendered_masks[best_idx].detach().cpu().numpy(),
                "name": f"batch{batchid}_idx{best_idx}"
            }

        # atualizar worst
        if errors[worst_idx] > global_worst_error:
            global_worst_error = errors[worst_idx]
            global_worst = {
                "img": (np.transpose(reg_images[worst_idx].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8),
                "kps": pred_keypoints2d_reproj[worst_idx].detach().cpu().numpy(),
                "seg": seg_masks[worst_idx].detach().cpu().numpy(),
                "render": rendered_masks[worst_idx].detach().cpu().numpy(),
                "name": f"batch{batchid}_idx{worst_idx}"
            }

    # --------------------------
    # Salvar best e worst global
    # --------------------------
    vis_folder = os.path.join(args.save_path, 'vis')
    train_vispath = os.path.join(vis_folder, "train")
    os.makedirs(train_vispath, exist_ok=True)

    def salvar_resultado(prefix, data):
        # imagem original com keypoints
        img_bgr = cv2.cvtColor(data["img"], cv2.COLOR_RGB2BGR)
        for kp in data["kps"]:
            cv2.circle(img_bgr, (int(kp[0]), int(kp[1])), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.imwrite(os.path.join(train_vispath, f"{prefix}_{data['name']}_keypoints.jpg"), img_bgr)

        # máscara da segNet
        seg_mask = (data["seg"].reshape(240, 320) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(train_vispath, f"{prefix}_{data['name']}_segnet.png"), seg_mask)

        # máscara renderizada pelo robô
        render_mask = (data["render"].reshape(240, 320) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(train_vispath, f"{prefix}_{data['name']}_robot.png"), render_mask)

    if global_best:
        salvar_resultado("best", global_best)

    if global_worst:
        salvar_resultado("worst", global_worst)

