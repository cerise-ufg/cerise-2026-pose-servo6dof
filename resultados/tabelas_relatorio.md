| Métrica (teste, 500 poses × 2 câmeras) | Transferência (pesos 2025) | Baseline (ImageNet) |
|---|---|---|
| MAE médio das juntas (°) | 12,68 | 13,51 |
| Erro 3D médio dos keypoints (mm) | 58,7 | 88,3 |
| Erro 3D mediano dos keypoints (mm) | 41,1 | 65,7 |
| AUC ADD 0–100 mm | 0,471 | 0,265 |
| Acurácia ADD @60 mm (%) | 68,0 | 32,8 |
| Acurácia ADD @100 mm (%) | 91,2 | 69,7 |
| Acurácia ADD @150 mm (%) ≈ 60 mm × 2,4 (escala do robô) | 96,7 | 90,2 |
| Acurácia ADD @175 mm (%) | 97,8 | 93,7 |
| Erro 2D médio dos keypoints (px) | 12,0 | 17,7 |
| Erro 2D mediano (px) | 7,9 | 11,8 |
| PCK@5 px por keypoint (%) | 30,1 | 19,9 |
| PCK@10 px por keypoint (%) | 61,7 | 43,6 |
| PCK@20 px por keypoint (%) | 87,1 | 71,4 |
| PCK@10 px por imagem (%) (critério 2025) | 57,1 | 26,8 |
| Épocas (melhor checkpoint na validação) | 20 | 12 |

| Junta | Transferência MAE (°) | Transferência mediana (°) | Baseline MAE (°) |
|---|---|---|---|
| joint_1 (base) | 4,19 | 2,94 | 7,20 |
| joint_2 (ombro) | 3,04 | 1,89 | 3,87 |
| joint_3 (cotovelo) | 3,91 | 2,54 | 5,41 |
| joint_4 (punho) | 10,52 | 6,64 | 9,25 |
| joint_5 (rotação da garra) | 41,73 | 42,49 | 41,83 |

| Câmera de teste | Modelo | MAE (°) | Erro 3D médio (mm) | AUC ADD | ADD@60 mm (%) | PCK@10 px (%) |
|---|---|---|---|---|---|---|
| frontal | Transferência (2 câmeras) | 12,70 | 57,4 | 0,486 | 69,0 | 58,8 |
| diagonal | Transferência (2 câmeras) | 12,65 | 60,0 | 0,457 | 67,0 | 64,5 |
| frontal | Baseline (2 câmeras) | 13,56 | 84,3 | 0,272 | 31,4 | 39,7 |
| diagonal | Baseline (2 câmeras) | 13,47 | 92,2 | 0,258 | 34,2 | 47,4 |
| frontal | Treino só frontal (xcam) | 23,08 | 140,4 | 0,084 | 8,6 | 22,2 |
| diagonal | Treino só frontal (xcam) | 32,57 | 418,5 | 0,000 | 0,0 | 3,5 |

**Transferência** — subconjunto fixo de 400 imagens do teste

| Perturbação | Severidade | Erro 3D médio (mm) | MAE (°) | PCK@10 px (%) | AUC ADD |
|---|---|---|---|---|---|
| (sem perturbação) | - | 57,4 | 12,41 | 64,7 | 0,479 |
| ruido_gaussiano | 0.02 | 69,8 | 13,27 | 57,9 | 0,375 |
| ruido_gaussiano | 0.05 | 99,0 | 15,64 | 41,7 | 0,211 |
| ruido_gaussiano | 0.1 | 213,2 | 23,57 | 15,0 | 0,026 |
| ruido_gaussiano | 0.2 | 544,4 | 36,69 | 3,5 | 0,000 |
| desfoque_movimento | 3 | 65,2 | 13,07 | 57,9 | 0,422 |
| desfoque_movimento | 7 | 96,0 | 15,21 | 41,3 | 0,225 |
| desfoque_movimento | 11 | 141,7 | 17,65 | 29,8 | 0,094 |
| desfoque_movimento | 15 | 206,6 | 20,44 | 20,2 | 0,031 |
| brilho | 0.4 | 60,5 | 12,68 | 62,8 | 0,453 |
| brilho | 0.6 | 58,1 | 12,53 | 64,2 | 0,474 |
| brilho | 1.4 | 57,2 | 12,37 | 64,5 | 0,480 |
| brilho | 1.8 | 57,9 | 12,46 | 63,8 | 0,479 |
| contraste | 0.3 | 67,2 | 13,24 | 58,3 | 0,391 |
| contraste | 0.6 | 59,2 | 12,65 | 63,9 | 0,461 |
| contraste | 1.5 | 57,3 | 12,41 | 64,6 | 0,480 |
| contraste | 2.0 | 57,9 | 12,44 | 63,9 | 0,476 |
| oclusao | 0.1 | 69,7 | 13,47 | 57,0 | 0,393 |
| oclusao | 0.2 | 80,8 | 14,57 | 49,3 | 0,324 |
| oclusao | 0.3 | 94,3 | 15,59 | 45,8 | 0,260 |
| jpeg | 50 | 76,3 | 13,89 | 54,7 | 0,332 |
| jpeg | 30 | 83,5 | 14,35 | 47,5 | 0,287 |
| jpeg | 15 | 113,8 | 16,19 | 32,7 | 0,150 |
| jpeg | 8 | 153,2 | 18,26 | 23,7 | 0,062 |
| resolucao | 0.5 | 65,7 | 13,21 | 58,0 | 0,414 |
| resolucao | 0.35 | 86,7 | 14,53 | 46,6 | 0,253 |
| resolucao | 0.25 | 109,2 | 15,49 | 39,0 | 0,154 |

**Baseline** — subconjunto fixo de 400 imagens do teste

| Perturbação | Severidade | Erro 3D médio (mm) | MAE (°) | PCK@10 px (%) | AUC ADD |
|---|---|---|---|---|---|
| (sem perturbação) | - | 88,2 | 13,25 | 43,5 | 0,263 |
| ruido_gaussiano | 0.02 | 90,6 | 13,64 | 41,8 | 0,250 |
| ruido_gaussiano | 0.05 | 109,7 | 14,97 | 34,9 | 0,172 |
| ruido_gaussiano | 0.1 | 169,7 | 19,46 | 22,1 | 0,069 |
| ruido_gaussiano | 0.2 | 512,4 | 35,15 | 4,1 | 0,001 |
| desfoque_movimento | 3 | 91,5 | 13,51 | 42,2 | 0,245 |
| desfoque_movimento | 7 | 113,0 | 14,76 | 34,9 | 0,153 |
| desfoque_movimento | 11 | 134,8 | 16,48 | 25,9 | 0,085 |
| desfoque_movimento | 15 | 162,1 | 18,42 | 19,2 | 0,045 |
| brilho | 0.4 | 93,7 | 13,48 | 42,4 | 0,242 |
| brilho | 0.6 | 90,4 | 13,34 | 42,9 | 0,255 |
| brilho | 1.4 | 87,6 | 13,25 | 43,7 | 0,264 |
| brilho | 1.8 | 87,7 | 13,29 | 43,2 | 0,261 |
| contraste | 0.3 | 93,1 | 13,85 | 40,0 | 0,243 |
| contraste | 0.6 | 89,0 | 13,36 | 43,2 | 0,261 |
| contraste | 1.5 | 87,8 | 13,22 | 43,7 | 0,263 |
| contraste | 2.0 | 87,9 | 13,23 | 43,3 | 0,262 |
| oclusao | 0.1 | 99,1 | 14,16 | 38,7 | 0,220 |
| oclusao | 0.2 | 106,9 | 15,03 | 36,2 | 0,188 |
| oclusao | 0.3 | 114,5 | 16,01 | 32,2 | 0,163 |
| jpeg | 50 | 121,1 | 15,17 | 30,2 | 0,132 |
| jpeg | 30 | 132,0 | 15,72 | 27,6 | 0,106 |
| jpeg | 15 | 214,6 | 19,23 | 14,6 | 0,058 |
| jpeg | 8 | 217,4 | 19,99 | 12,8 | 0,029 |
| resolucao | 0.5 | 97,7 | 13,80 | 39,5 | 0,215 |
| resolucao | 0.35 | 113,9 | 14,75 | 32,8 | 0,151 |
| resolucao | 0.25 | 130,0 | 15,88 | 28,1 | 0,101 |

| Etapa (batch 1, GB10) | Média (ms) | Mediana (ms) | p95 (ms) |
|---|---|---|---|
| Depth Anything V2 (ViT-L, 518 px) | 138,9 | 136,4 | 179,2 |
| HPE (DepthNet + integral + regressão + FK) | 74,9 | 71,0 | 99,6 |
| Total | 213,8 | 201,7 | 270,3 |

FPS total: 4,68; FPS só HPE: 13,34; memória de pico: 8,04 GB
