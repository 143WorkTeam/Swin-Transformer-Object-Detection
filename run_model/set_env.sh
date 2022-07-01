# install apex
# check this issue https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/issues/113
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# install mmcv
pip install mmcv-full==1.4.0
# set up
pip install -r requirements/build.txt
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install -v -e .  # or "python setup.py develop"

# build wheels
python -m pip wheel mmcv-full==1.4.0
