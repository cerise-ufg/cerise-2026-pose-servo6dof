"""Geometria do braco servo6dof e das cameras do estudio_hd (CERISE 2026, frente POSE).

- Cinematica direta (numpy e torch/diferenciavel) lida do URDF gerado pelo xacro.
- Keypoints (7): origem da base + origens dos frames filhos de joint_1..joint_5 + ponta da garra.
- Cameras: poses do mundo estudio_hd.sdf (convencao Gazebo: camera olha ao longo de +X do link,
  Z para cima), convertidas para o frame otico (OpenCV: X direita, Y baixo, Z para frente).
"""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
URDF_PADRAO = os.path.join(AQUI, "servo6dof.urdf")

JUNTAS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
KEYPOINTS = ["base", "junta_1", "junta_2", "junta_3", "junta_4", "junta_5", "ponta_garra"]
# esqueleto para desenho (pares de indices de KEYPOINTS)
ARESTAS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
# ponta da garra no frame base_garra (filho de joint_5): ponto medio entre as origens das duas
# pincas (joint_pinca_a / joint_pinca_b) com a garra em joint_6 = 0  ->  (-0.202+0.202)/2, 0.12+0.13, 0.01
PONTA_GARRA_OFFSET = np.array([0.0, 0.25, 0.01])

# ---------------- cameras (estudio_hd.sdf) ----------------
W, H = 1280, 720
HFOV = 1.0472
FX = (W / 2.0) / math.tan(HFOV / 2.0)  # ~1108.5
FY = FX
CX, CY = W / 2.0, H / 2.0
CAMERAS_SDF = {
    # x y z roll pitch yaw (pose do <model>; o <link> e o <sensor> nao tem <pose> propria no SDF)
    "frontal": (3.2, 0.4, 1.6, 0.0, 0.215, 3.14159),
    "diagonal": (2.6, 2.6, 2.4, 0.0, 0.414, -2.439),
}
# colunas = eixos do frame otico escritos no frame do link Gazebo
R_LINK_OPTICO = np.array([[0.0, 0.0, 1.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0]])


def K_matrix():
    return np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])


def rpy_to_R(r, p, y):
    """Convencao URDF/SDF: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def T_from(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_mundo_camera_otica(nome):
    x, y, z, r, p, yw = CAMERAS_SDF[nome]
    T_w_link = T_from(rpy_to_R(r, p, yw), [x, y, z])
    return T_w_link @ T_from(R_LINK_OPTICO, [0, 0, 0])


def T_camera_mundo(nome):
    """Transformacao mundo -> frame otico da camera (o robo esta na origem do mundo, z=0)."""
    return np.linalg.inv(T_mundo_camera_otica(nome))


def projetar(K, pts_cam):
    pts_cam = np.asarray(pts_cam)
    uvw = pts_cam @ K.T
    return uvw[..., :2] / uvw[..., 2:3]


# ---------------- cinematica (URDF) ----------------
def ler_cadeia(urdf_path=URDF_PADRAO):
    """Le a cadeia world -> base_link -> ... -> base_garra do URDF."""
    root = ET.parse(urdf_path).getroot()
    juntas = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        a = j.find("axis")
        juntas[j.find("child").get("link")] = dict(
            nome=j.get("name"), tipo=j.get("type"), pai=j.find("parent").get("link"),
            xyz=np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]),
            rpy=np.array([float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()]),
            eixo=np.array([float(v) for v in (a.get("xyz") if a is not None else "1 0 0").split()]))
    cadeia = []
    link = "base_garra"
    while link in juntas:
        cadeia.append(juntas[link])
        link = juntas[link]["pai"]
    cadeia = cadeia[::-1]  # virtual_joint, joint_1..joint_5
    nomes = [c["nome"] for c in cadeia]
    assert nomes == ["virtual_joint"] + JUNTAS, nomes
    return cadeia


def axis_angle_R(eixo, ang):
    eixo = eixo / np.linalg.norm(eixo)
    x, y, z = eixo
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def fk_keypoints_np(q, cadeia):
    """q: (5,) rad -> (7,3) keypoints no frame do mundo (= frame da base)."""
    T = np.eye(4)
    kps = []
    qi = 0
    for c in cadeia:
        T = T @ T_from(rpy_to_R(*c["rpy"]), c["xyz"])
        if c["tipo"] == "revolute":
            T = T @ T_from(axis_angle_R(c["eixo"], q[qi]), [0, 0, 0])
            qi += 1
        kps.append(T[:3, 3].copy())
    kps.append((T @ np.append(PONTA_GARRA_OFFSET, 1.0))[:3])
    return np.stack(kps)  # base, j1..j5, ponta


class FKTorch:
    """Cinematica direta em lote, diferenciavel (usada pelo modelo HPE portado)."""

    def __init__(self, urdf_path=URDF_PADRAO, device="cuda"):
        import torch
        self.torch = torch
        cadeia = ler_cadeia(urdf_path)
        self.fixas = torch.tensor(np.stack([T_from(rpy_to_R(*c["rpy"]), c["xyz"]) for c in cadeia]),
                                  dtype=torch.float32, device=device)  # (6,4,4)
        self.eixos = torch.tensor(np.stack([c["eixo"] / np.linalg.norm(c["eixo"]) for c in cadeia[1:]]),
                                  dtype=torch.float32, device=device)  # (5,3)
        self.ponta = torch.tensor(np.append(PONTA_GARRA_OFFSET, 1.0), dtype=torch.float32, device=device)

    def _rot(self, eixo, ang):
        torch = self.torch
        x, y, z = eixo[0], eixo[1], eixo[2]
        c, s = torch.cos(ang), torch.sin(ang)
        C = 1 - c
        R = torch.stack([
            torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], -1),
            torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], -1),
            torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1)], -2)
        return R  # (B,3,3)

    def link_transforms(self, q):
        """q: (B,5) -> (B,7,4,4): T do mundo para [base, elo1..elo5, ponta(frame base_garra)]"""
        torch = self.torch
        B = q.shape[0]
        dev = q.device
        T = self.fixas[0].to(dev).unsqueeze(0).expand(B, 4, 4)
        Ts = [T]
        for i in range(5):
            Tj = torch.eye(4, device=dev).unsqueeze(0).repeat(B, 1, 1)
            Tj[:, :3, :3] = self._rot(self.eixos[i].to(dev), q[:, i])
            T = T @ self.fixas[i + 1].to(dev) @ Tj
            Ts.append(T)
        Tp = T.clone()
        Tp[:, :, 3] = T @ self.ponta.to(dev)
        Ts.append(Tp)
        return torch.stack(Ts, 1)

    def keypoints(self, q):
        return self.link_transforms(q)[:, :, :3, 3]  # (B,7,3)
