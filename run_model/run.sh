# setup env
export MMDET_DATASETS=/mnt/coco/
export OMP_NUM_THREADS=8

mkdir checkpoints
cd checkpoints
wget https://github.91chi.fun/https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.pth
wget https://github.91chi.fun/https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
# OR
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.pth
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth

# Test on existing model and checkpoint
# refer for the website https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md
# weightes are downloaded from https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn and save into checkpoints
python tools/test.py\
 configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py \
 checkpoints/cascade_mask_rcnn_swin_base_patch4_window7.pth \
 --show \
 --work-dir=results \
 --eval="bbox" \
# train on smaller files for better debug
python -m exps.make_smaller_coco

python tools/train.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=12 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=12 \
                    log_config.interval=100 \
                    runner.max_epochs=3 \
                    data.train.ann_file='/mnt/coco/annotations/instances_train2017.small.json' \
    --work-dir=cps
# run all
python tools/train.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=12 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=12 \
                    log_config.interval=300 \
                    runner.max_epochs=2 \
    --work-dir=cps


    



