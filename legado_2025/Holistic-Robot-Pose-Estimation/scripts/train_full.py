import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from lib.core.function import farward_loss, validate
from lib.dataset.const import INITIAL_JOINT_ANGLE
from lib.models.full_net import get_rootNetwithRegInt_model
from lib.utils.urdf_robot import URDFRobot
from lib.utils.utils import set_random_seed, create_logger, get_dataloaders, get_scheduler, resume_run, save_checkpoint
from torchnet.meter import AverageValueMeter
from tqdm import tqdm
from lib.utils.metrics import summary_add_pck
    
def train_full(args):
    
    torch.autograd.set_detect_anomaly(True)
    set_random_seed(808)
    
    save_folder, ckpt_folder, log_folder, writer = create_logger(args)
    ckpt_folder = "experiments/pandareal_realsense_full/ckpt"
    urdf_robot_name = args.urdf_robot_name
    robot = URDFRobot(urdf_robot_name)
 
    device_id = args.device_id
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    ds_iter_train, test_loader_dict = get_dataloaders(args)
    
    init_param_dict = {
        "robot_type" : urdf_robot_name,
        "pose_params": INITIAL_JOINT_ANGLE,
        "cam_params": np.eye(4,dtype=float),
        "init_pose_from_mean": True
    }
    import os
    if args.use_rootnet_with_reg_int_shared_backbone:
        print("regression and integral shared backbone, with rootnet 2 backbones in total")
        model = get_rootNetwithRegInt_model(init_param_dict, args)
    else:
        assert 0
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    curr_max_auc = 0.0
    curr_max_auc_4real = { "realsense": 0.0 }
    start_epoch, last_epoch, end_epoch = 0, -1, args.n_epochs
    if args.resume_run:
        start_epoch, last_epoch, curr_max_auc, curr_max_auc_4real = resume_run(args, model, optimizer, device)
        
    lr_scheduler = get_scheduler(args, optimizer, last_epoch)

    import csv
    import os
    print("Resume")
    model.load_state_dict(torch.load(os.path.join("experiments", "pandareal_realsense_full/ckpt/checkpoint.pk"))["model_state_dict"])
    csv_path = os.path.join(ckpt_folder, "real_training_log.csv")

    best_auc_adds = {"photo": 0.0, "dr": 0.0, "realsense": 0.0}
    def log_metrics_to_csv(csv_file_name, epoch, metrics, add_thresholds, pck_thresholds, robot,
                           dsnames=["_train", "_photo", "_dr", "_realsense"]):
        """
        Logs validation metrics to a CSV file. Creates the header if the file does not exist.
        """

        # 1. Define os nomes das métricas principais
        metric_names = [
            'loss', 'pose_loss', 'rot_loss', 'rot_diff', 'trans_loss', 'uv_loss',
            'depth_loss', 'error2d_loss', 'error3d_loss', 'error2d_int_loss',
            'error3d_int_loss', 'error3d_align_loss', 'mean_joint_error',
            'AUC_ADD', 'AUC_PCK', 'AUC_ADD_integral_xyz_metrics', 'AUC_PCK_integral_xyz_metrics'
        ]

        # 2. Cria os cabeçalhos dinamicamente, adicionando dsname
        headers = ['epoch']
        for dsname in dsnames:
            # Métricas principais
            headers.extend([f"Val/{name}{dsname}" for name in metric_names])
            # Thresholds
            headers.extend([f"Val/ADD_{k}_mm{dsname}" for k in add_thresholds])
            headers.extend([f"Val/PCK_{k}_pixel{dsname}" for k in pck_thresholds])
            headers.extend([f"Val/ADD_{k}_mm_integral_xyz_metrics{dsname}" for k in add_thresholds])
            headers.extend([f"Val/PCK_{k}_pixel_integral_xyz_metrics{dsname}" for k in pck_thresholds])
            # Keypoints
            headers.extend([f"Val/distance3D_keypoint_{k + 1}{dsname}" for k in range(len(robot.link_names))])
            headers.extend([f"Val/distance2D_keypoint_{k + 1}{dsname}" for k in range(len(robot.link_names))])
            headers.extend([f"Val/distance3D_keypoint_{k + 1}_integral_xyz_metrics{dsname}" for k in
                            range(len(robot.link_names))])
            headers.extend([f"Val/distance2D_keypoint_{k + 1}_integral_xyz_metrics{dsname}" for k in
                            range(len(robot.link_names))])
            # Joints
            headers.extend([f"Val/l1error_joint_{k + 1}{dsname}" for k in range(robot.dof)])

        # 3. Cria arquivo se não existir
        file_exists = os.path.isfile(csv_file_name)
        with open(csv_file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(headers)

            # 4. Cria a linha de dados
            row = [epoch]
            for dsname in dsnames:
                # Métricas principais
                for name in metric_names:
                    val = metrics[f'{name}{dsname}'] if f'{name}{dsname}' in metrics else metrics.get(name, 0.0)
                    row.append(val if not hasattr(val, "mean") else val.mean())

                # Thresholds
                for k in add_thresholds:
                    key = f'ADD_{k}_mm{dsname}'
                    val = metrics[key]
                    row.append(val if not hasattr(val, "mean") else val.mean())
                for k in pck_thresholds:
                    key = f'PCK_{k}_pixel{dsname}'
                    val = metrics[key]
                    row.append(val if not hasattr(val, "mean") else val.mean())
                for k in add_thresholds:
                    key = f'ADD_{k}_mm_integral_xyz_metrics{dsname}'
                    val = metrics[key]
                    row.append(val if not hasattr(val, "mean") else val.mean())
                for k in pck_thresholds:
                    key = f'PCK_{k}_pixel_integral_xyz_metrics{dsname}'
                    val = metrics[key]
                    row.append(val if not hasattr(val, "mean") else val.mean())

                # Keypoints
                for k in range(len(robot.link_names)):
                    key = f'distance3D_keypoint_{k + 1}{dsname}'
                    val = metrics[key].mean
                    row.append(val)
                for k in range(len(robot.link_names)):
                    key = f'distance2D_keypoint_{k + 1}{dsname}'
                    val = metrics[key].mean
                    row.append(val)
                for k in range(len(robot.link_names)):
                    key = f'distance3D_keypoint_{k + 1}_integral_xyz_metrics{dsname}'
                    val = metrics[key].mean
                    row.append(val)
                for k in range(len(robot.link_names)):
                    key = f'distance2D_keypoint_{k + 1}_integral_xyz_metrics{dsname}'
                    val = metrics[key].mean
                    row.append(val)

                # Joints
                for k in range(robot.dof):
                    key = f'l1error_joint_{k + 1}{dsname}'
                    val = metrics[key].mean
                    row.append(val)

            # 5. Escreve a linha
            writer.writerow(row)
    from collections import defaultdict

    for epoch in range(start_epoch, end_epoch + 1):
        print('In epoch {},  script: full network training (JointNet/RotationNet/KeypoinNet/DepthNet)'.format(epoch + 1))
        model.train()
        iterator = tqdm(ds_iter_train, dynamic_ncols=True)
        # Inicialize todos os AverageValueMeters

        # Inicialize todos os AverageValueMeters
        losses_pose, losses_rot, losses_trans, losses_uv, losses_depth, losses_error2d, losses_error3d, losses_error2d_int, losses_error3d_int, losses_error3d_align, rotation_diff = \
            AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter()
        loss_train = AverageValueMeter()

        alldis = defaultdict(list)
        alldis_int = defaultdict(list)

        # Mantenha os thresholds e AverageValueMeters por keypoint e joint
        add_thresholds = [1, 5, 10, 20, 40, 60, 80, 100]
        pck_thresholds = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
        metric_dis3d = [AverageValueMeter() for _ in range(len(robot.link_names))]
        metric_dis2d = [AverageValueMeter() for _ in range(len(robot.link_names))]
        metric_dis3d_int = [AverageValueMeter() for _ in range(len(robot.link_names))]
        metric_dis2d_int = [AverageValueMeter() for _ in range(len(robot.link_names))]
        metric_l1joint = [AverageValueMeter() for _ in range(robot.dof)]

        for batchid, sample in enumerate(iterator):
            optimizer.zero_grad()
            loss, loss_dict, metric_dict = farward_loss(args=args, input_batch=sample, model=model, robot=robot, device=device, device_id=device_id, train=True)
            loss.backward()
            clipping_value = args.clip_gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step()
            optimizer.step()

            # Adiciona os valores aos AverageValueMeters
            loss_train.add(loss.detach().cpu().numpy())
            losses_pose.add(loss_dict["loss_joint"].detach().cpu().numpy())
            losses_rot.add(loss_dict["loss_rot"].detach().cpu().numpy())
            losses_trans.add(loss_dict["loss_trans"].detach().cpu().numpy())
            losses_depth.add(loss_dict["loss_depth"].detach().cpu().numpy())
            losses_uv.add(loss_dict["loss_uv"].detach().cpu().numpy())
            losses_error2d.add(loss_dict["loss_error2d"].detach().cpu().numpy())
            losses_error3d.add(loss_dict["loss_error3d"].detach().cpu().numpy())
            losses_error2d_int.add(loss_dict["loss_error2d_int"].detach().cpu().numpy())
            losses_error3d_int.add(loss_dict["loss_error3d_int"].detach().cpu().numpy())
            losses_error3d_align.add(loss_dict["loss_error3d_align"].detach().cpu().numpy())

            # Estende as listas de distâncias para cálculo posterior de PCK e ADD
            alldis["dis3d"].extend(metric_dict["image_dis3d_avg"])
            alldis["dis2d"].extend(metric_dict["image_dis2d_avg"]) 
            alldis["jointerror"].extend(metric_dict["image_l1jointerror_avg"])
            alldis_int["dis3d"].extend(metric_dict["image_dis3d_avg_int"])
            alldis_int["dis2d"].extend(metric_dict["image_dis2d_avg_int"])

            rotation_diff.add(metric_dict["rotation_diff"].detach().cpu().numpy())

            # Adiciona os valores por keypoint e joint
            for id in range(len(robot.link_names)):
                metric_dis3d[id].add(metric_dict["batch_dis3d_avg"][id])
                metric_dis2d[id].add(metric_dict["batch_dis2d_avg"][id])
                metric_dis3d_int[id].add(metric_dict["batch_dis3d_avg_int"][id])
                metric_dis2d_int[id].add(metric_dict["batch_dis2d_avg_int"][id])
            for id in range(robot.dof):
                metric_l1joint[id].add(metric_dict["batch_l1jointerror_avg"][id])

            # Calcula os summaries de PCK e ADD
        summary = summary_add_pck(alldis, add_thresholds, pck_thresholds)
        summary_int = summary_add_pck(alldis_int, add_thresholds, pck_thresholds)

        # Constrói o dicionário de métricas completo
        all_metrics = {
            'loss': loss_train.mean,
            'pose_loss': losses_pose.mean,
            'rot_loss': losses_rot.mean,
            'trans_loss': losses_trans.mean,
            'uv_loss': losses_uv.mean,
            'depth_loss': losses_depth.mean,
            'error2d_loss': losses_error2d.mean,
            'error3d_loss': losses_error3d.mean,
            'error2d_int_loss': losses_error2d_int.mean,
            'error3d_int_loss': losses_error3d_int.mean,
            'error3d_align_loss': losses_error3d_align.mean,
            'rot_diff': rotation_diff.mean,
            'mean_joint_error': np.mean(alldis["jointerror"]) / np.pi * 180.0,
            'AUC_ADD': summary['ADD/AUC'],
            'AUC_PCK': summary['PCK/AUC'],
            'AUC_ADD_integral_xyz_metrics': summary_int['ADD/AUC'],
            'AUC_PCK_integral_xyz_metrics': summary_int['PCK/AUC'],
        }

        # Adiciona as métricas por threshold
        for k in add_thresholds:
            all_metrics[f'ADD_{k}_mm'] = summary[f'ADD_{k}_mm']
        for k in pck_thresholds:
            all_metrics[f'PCK_{k}_pixel'] = summary[f'PCK_{k}_pixel']
        for k in add_thresholds:
            all_metrics[f'ADD_{k}_mm_integral_xyz_metrics'] = summary_int[f'ADD_{k}_mm']
        for k in pck_thresholds:
            all_metrics[f'PCK_{k}_pixel_integral_xyz_metrics'] = summary_int[f'PCK_{k}_pixel']

        # Adiciona as métricas por keypoint e joint
        for k in range(len(robot.link_names)):
            all_metrics[f'distance3D_keypoint_{k + 1}'] = metric_dis3d[k]
            all_metrics[f'distance2D_keypoint_{k + 1}'] = metric_dis2d[k]
            all_metrics[f'distance3D_keypoint_{k + 1}_integral_xyz_metrics'] = metric_dis3d_int[k]
            all_metrics[f'distance2D_keypoint_{k + 1}_integral_xyz_metrics'] = metric_dis2d_int[k]
        for k in range(robot.dof):
            all_metrics[f'l1error_joint_{k + 1}'] = metric_l1joint[k]

        dsname = "_train"  # ou qualquer string que queira adicionar
        all_metrics = {k + dsname: v for k, v in all_metrics.items()}
        # Validação
        auc_adds = {}
        for dsname, loader in test_loader_dict.items():
            auc_add, all_metrics_val = validate(arg=args, epoch=epoch, dsname=dsname, loader=loader, model=model,
                robot=robot, device=device, device_id=device_id)
            auc_adds[dsname] = auc_add
            all_metrics = all_metrics | all_metrics_val

        log_metrics_to_csv(csv_path, epoch, all_metrics, add_thresholds, pck_thresholds, robot)

        for key, value in auc_adds.items():
            if value > best_auc_adds[key]:
                best_auc_adds[key] = value
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    ckpt_folder="/content/drive/MyDrive/Mestrado/Holistic-Robot-Pose-Estimation/experiments/pandareal_realsense_full/ckpt",
                    epoch=epoch,
                    lr_scheduler=lr_scheduler,
                    save_path=f"/content/drive/MyDrive/Mestrado/Holistic-Robot-Pose-Estimation/experiments/pandareal_realsense_full/ckpt/best_model_auc{key}.pk",
                    all_metrics=all_metrics
                )
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                ckpt_folder="/content/drive/MyDrive/Mestrado/Holistic-Robot-Pose-Estimation/experiments/pandareal_realsense_full/ckpt",
                epoch=epoch,
                lr_scheduler=lr_scheduler,
                save_path="/content/drive/MyDrive/Mestrado/Holistic-Robot-Pose-Estimation/experiments/pandareal_realsense_full/ckpt/checkpoint.pk",
                all_metrics=all_metrics
            )

    print("Training Finished!")
