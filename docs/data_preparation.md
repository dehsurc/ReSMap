# Data Preparation

## nuScenes

**Step 1.** Download the [nuScenes](https://www.nuscenes.org/download) full dataset (v1.0-trainval)
and the map expansion pack (v1.3). Unzip the map expansion into `maps/`.

**Step 2.** Generate the annotation files for the geographically disjoint split. They are written to
the data root.

```bash
python tools/data_converter/nuscenes_converter.py --data-root ./datasets/nuscenes --newsplit
```

## AID4AD satellite images

**Step 3.** Download [AID4AD](https://github.com/DriverlessMobility/AID4AD) and export one aerial view
per nuScenes sample with its scripts. Each view is a 400 × 200 RGB image (0.15 m/px) covering
60 m × 30 m around the ego pose, saved as `AID4AD_frames/{location}/{counter}_{sample_token}.png`.

The AID4AD imagery comes from Google Earth and is subject to the Google Earth Terms of Service, so it is
not redistributed here. Please obtain it from the official AID4AD release.

## Folder structure

Put (or symlink) everything under `./datasets/nuscenes`:

```
ReSMap
├── configs/
├── plugin/
├── tools/
├── datasets/
│   └── nuscenes/
│       ├── maps/
│       │   ├── expansion/
│       │   └── ...
│       ├── samples/
│       ├── v1.0-trainval/
│       ├── AID4AD_frames/
│       │   ├── boston-seaport/
│       │   ├── singapore-hollandvillage/
│       │   ├── singapore-onenorth/
│       │   └── singapore-queenstown/
│       ├── nuscenes_map_infos_train_newsplit.pkl
│       └── nuscenes_map_infos_val_newsplit.pkl
```

The ground-truth vectors are generated on the fly from the nuScenes map with MapTracker's processing
code. The evaluator builds the validation ground truth once and caches it under `.gt_cache/`.
