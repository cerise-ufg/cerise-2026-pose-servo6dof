"""Interface de robo do HPE (mesma API do URDFRobot de 2025) para o servo6dof.

O URDFRobot original (lib/utils/urdf_robot.py) usava urdfpytorch + caminhos fixos do Panda/Kuka/Baxter
e pyrender. Aqui a cinematica direta e a da cadeia joint_1..joint_5 do URDF do servo6dof (FKTorch,
diferenciavel), com 7 keypoints (base, frames de joint_1..joint_5, ponta da garra), mantendo os metodos
usados pelo modelo e pelas metricas: get_keypoints, get_keypoints_root, get_rotation_at_specific_root,
get_keypoints_only_fk, get_keypoints_only_fk_at_specific_root.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from servo6dof.geometria import FKTorch, KEYPOINTS, JUNTAS, URDF_PADRAO  # noqa: E402
from utils.geometries import quat_to_rotmat, rot6d_to_rotmat, rot9d_to_rotmat, rotmat_to_quat, rotmat_to_rot6d  # noqa: E402


class Servo6dofRobot:
    def __init__(self, urdf_path=URDF_PADRAO, device="cuda"):
        self.robot_type = "servo6dof"
        self.dof = len(JUNTAS)
        self.link_names = list(KEYPOINTS)
        self.actuated_joint_names = list(JUNTAS)
        self.fk = FKTorch(urdf_path, device=device)

    @staticmethod
    def _rotmat(rot):
        if rot.shape[1] == 6:
            return rot6d_to_rotmat(rot)
        if rot.shape[1] == 4:
            return quat_to_rotmat(rot)
        if rot.shape[1] == 9:
            return rot9d_to_rotmat(rot)
        raise NotImplementedError

    def _b2c(self, rot, trans):
        B = rot.shape[0]
        T = torch.zeros((B, 4, 4), dtype=torch.float32, device=trans.device)
        T[:, :3, :3] = self._rotmat(rot).to(trans.device)
        T[:, :3, 3] = trans
        T[:, 3, 3] = 1.0
        return T.unsqueeze(1)

    def get_TWL(self, cfgs):
        return self.fk.link_transforms(cfgs.float())

    def get_keypoints(self, jointcfgs, b2c_rot, b2c_trans):
        TWL = self._b2c(b2c_rot, b2c_trans) @ self.get_TWL(jointcfgs)
        return TWL[:, :, :3, 3]

    def get_keypoints_root(self, jointcfgs, b2c_rot, b2c_trans, root=0):
        if root == 0:
            return self.get_keypoints(jointcfgs, b2c_rot, b2c_trans)
        TWL_base = self.get_TWL(jointcfgs)
        TWL_root_inv = torch.linalg.inv(TWL_base[:, root:root + 1])
        TWL = self._b2c(b2c_rot, b2c_trans) @ (TWL_root_inv @ TWL_base)
        return TWL[:, :, :3, 3]

    def get_rotation_at_specific_root(self, jointcfgs, b2c_rot, b2c_trans, root=0):
        if root == 0:
            return b2c_rot
        TWL = self._b2c(b2c_rot, b2c_trans) @ self.get_TWL(jointcfgs)
        R = TWL[:, root, :3, :3]
        return rotmat_to_rot6d(R) if b2c_rot.shape[1] == 6 else rotmat_to_quat(R)

    def get_keypoints_only_fk(self, jointcfgs):
        return self.get_TWL(jointcfgs)[:, :, :3, 3]

    def get_keypoints_only_fk_at_specific_root(self, jointcfgs, root=0):
        TWL_base = self.get_TWL(jointcfgs)
        if root == 0:
            return TWL_base[:, :, :3, 3]
        return (torch.linalg.inv(TWL_base[:, root:root + 1]) @ TWL_base)[:, :, :3, 3]
