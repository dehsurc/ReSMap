# Installation

The code is tested with Python 3.8, CUDA 11.6, PyTorch 1.12.1, mmcv-full 1.6.0, mmdet 2.28.2,
mmsegmentation 0.30.0 and mmdet3d 1.0.0rc6.

**Step 1.** Create a conda environment and activate it.

```bash
conda create --name resmap python=3.8 -y
conda activate resmap
```

**Step 2.** Install PyTorch.

```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

**Step 3.** Install the MMCV series.

```bash
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
```

**Step 4.** Install mmdet3d from source.

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git -b v1.0.0rc6
cd mmdetection3d
pip install -v -e .
cd ..
```

**Step 5.** Install the remaining requirements.

```bash
pip install -r requirements.txt
```

The ground-truth geometry is built with Shapely, and the pinned version (1.8.5) is the one the reported
numbers were evaluated with.
