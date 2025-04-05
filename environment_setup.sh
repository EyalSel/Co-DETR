# Create and activate environment
conda create -n co-detr-py310 python=3.10.16
conda activate co-detr-py310

# Install PyTorch ecosystem
conda install cudatoolkit=11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install MMCV (downloaded and installed specific wheel)
wget https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/mmcv_full-1.5.0-cp310-cp310-manylinux1_x86_64.whl
pip install mmcv_full-1.5.0-cp310-cp310-manylinux1_x86_64.whl

# Install basic utilities
pip install ipython gpustat
pip install tqdm
pip install google-cloud-storage

# Install dependencies for the model
pip install terminaltables
pip install pycocotools
pip install fairscale==0.4.6
pip install git+https://github.com/facebookresearch/fvcore.git
pip install scipy==1.7.3
pip install timm
pip install einops
pip install numpy==1.21.6
pip install matplotlib==3.5.3
pip install opencv-python==4.8.1.78
