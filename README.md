# Robust Scene Flow via Task-Wise Adaptation of Pretrained Stereo and Optical Flow

Code for our RoCo-Spring scene flow entry (Track 3). Scene flow is estimated from two
consecutive stereo pairs by fine-tuning a pretrained stereo network
([DEFOM-Stereo](https://github.com/Insta360-Research-Team/DEFOM-Stereo)) and a pretrained
optical flow network ([DPFlow](https://github.com/hmorimitsu/ptlflow)) separately, and composing
their outputs:

```
d1 = S(L_t, R_t)        dn = S(L_t+1, R_t+1)        f = F(L_t, L_t+1)
d2(p) = dn(p + f(p))    (bilinear sampling)
```

Training uses quadruplet-level appearance augmentation (Q4Aug), ground-truth-guided transport of
spatially localized corruptions, and correspondence-guided masking (CorrMask).

## Results

Official [Spring scene flow benchmark](https://spring-benchmark.org/sceneflow) entry
**RoCo-19-Q4-Independent**:

| 1px total | SF | 1px D1 | 1px D2 | 1px Fl |
|---|---|---|---|---|
| 12.200 | 4.780 | 8.759 | 9.388 | 3.295 |

Robustness columns of the same leaderboard:

| Δ1px D1 | ΔAbs D1 | ΔD1 | Δ1px D2 | ΔAbs D2 | ΔD2 | ΔEPE Fl | ΔFl Fl | Δ1px Fl |
|---|---|---|---|---|---|---|---|---|
| 23.330 | 3.840 | 4.502 | 23.632 | 3.925 | 7.828 | 1.976 | 6.774 | 13.305 |

See the leaderboard for the per-region breakdown and metric definitions.

## Setup

```bash
pip install -r requirements.txt
bash setup_third_party.sh        # clones DEFOM-Stereo (+ a 3-line patch) and flow_library
```

The script checks out the commits used for the paper (DEFOM-Stereo `5b27591`, flow_library
`8454aed`). DPFlow comes from `ptlflow==0.4.2`.

The patch lets DEFOM-Stereo keep the gradient through its recurrent disparity state
(`patches/defom_stereo_detach.patch`); the default behaviour is unchanged.

Pretrained initializations go into `checkpoints/`:

```
checkpoints/defom-stereo/defomstereo_vits_sceneflow.pth   # DEFOM-Stereo ViT-S, Scene Flow (from the DEFOM-Stereo repo)
checkpoints/dpflow/dpflow-spring-69bac7fa.ckpt              # https://github.com/hmorimitsu/ptlflow/releases/download/weights1/dpflow-spring-69bac7fa.ckpt
```

Spring goes into `data/spring` (`train/` and `test/` as distributed by
[spring-benchmark.org](https://spring-benchmark.org)). All three locations can be changed with
`ROCO_CHECKPOINTS`, `ROCO_SPRING` and `ROCO_THIRD_PARTY`.

## Training

```bash
python train.py --out runs/final --experiment final
```

With two visible GPUs the stereo and flow networks are placed on separate devices; with one GPU
both share it. Scenes 0038, 0039, 0041, 0043, 0044, 0045 and 0047 are held out
(`configs/spring_holdout.json`); the remaining 30 scenes give 4,024 training quadruplets.

Main settings: 320x640 crops, AdamW (lr 3e-6 for DEFOM-Stereo, 3e-7 for its monocular
encoder/decoder, 1e-6 for DPFlow, weight decay 1e-5), gradient clipping at 1 per network, 8
DEFOM-Stereo iterations in training and 16 at test time, seed 9610. The model is validated every
2,012 updates on 49 held-out pairs plus four synthetic corruptions; the checkpoint with the lowest

```
J = 0.5 * (N(clean) + mean_c N(c)),   N = D1Abs/7.466 + D2Abs/7.5935 + FlowEPE/2.527
```

is kept as `best.pt`. Training stops after at most 10 epochs, or earlier after 6 validations
without improvement (minimum 3 epochs).

Ablations in the paper use `--experiment`:

| `--experiment` | Change from `final` |
|---|---|
| `final` | submitted configuration: independent losses, Q4Aug + transport + CorrMask |
| `geometry` | shared resize/crop only, no appearance augmentation |
| `independent_view` | appearance corruption sampled separately for each view |
| `q4aug` | Q4Aug with spatial masks placed at the same coordinates in all views |
| `q4aug_transport` | Q4Aug with ground-truth transport, no CorrMask |
| `random_mask` | CorrMask centred at a uniformly random location |
| `stereo_only` | DPFlow kept fixed |
| `flow_only` | DEFOM-Stereo kept fixed |
| `coupled` | adds `lambda_D2 * L_D2` on the composed disparity (`--lambda-d2`, default 1.0) |

## Inference

```bash
python predict.py --images data/spring/test --out predictions/clean --checkpoint runs/final/best.pt
```

This writes disparity, forward/backward flow and d2 for both cameras in the Spring file layout.
Pack the folder into the benchmark's HDF5 format with the official Spring subsampling tools.

## Analysis on held-out frames

```bash
python tools/build_frozen_pairs.py                 # 98 pairs not used for checkpoint selection
python tools/frozen_eval.py --tag final --checkpoint runs/final/best.pt
python tools/frozen_eval.py --tag final --checkpoint runs/final/best.pt \
    --protocol consistent --families rain snow spatter --no-clean
```

`--protocol independent` corrupts each view separately; `--protocol consistent` places one
rain/snow/spatter mask in the left reference view and moves it to the other views with the
ground-truth disparity and flow. These pairs come from the same scenes as the validation pairs, so
they are not an unseen-scene test set.

Confidence intervals for differences between models (10,000 scene-clustered bootstrap replicates
over the 7 held-out scenes):

```bash
python tools/bootstrap_ci.py --results results/frozen --compare final:q4aug_transport --conditions all
python tools/bootstrap_ci.py --results results/frozen_consistent --compare final:independent_view \
    --conditions spatial
```

Agreement between the D2 gradient and each network's own loss gradient, measured on a fixed list
of 48 training quadruplets without updating the model:

```bash
python tools/gradient_alignment.py --checkpoint final=runs/final/best.pt \
    --checkpoint coupled=runs/coupled/best.pt --out results/gradient_alignment.csv
```

## License

Code written for this repository: CC BY-NC-SA 4.0 (`LICENSE`). Non-commercial use only. External
code, checkpoints and datasets are not redistributed; their terms are listed in
`THIRD_PARTY_NOTICES.md`. Note that the DPFlow pretrained weights are released by their authors for
academic and research use only.
