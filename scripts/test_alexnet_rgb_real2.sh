set -x
tr_set='trainval'
test_set='test'

modality="images"
  PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net2.py --gpu 1 \
    --def scripts/alexnet_rgb_real/test.prototxt \
    --net /nfs.yoda/xiaolonw/fast_rcnn/fast-rcnn-distillation/output/alexnet_rgb_alexnet_hha/nyud2_images+hha_2015_trainval/fast_rcnn_iter_40000_rgb.caffemodel \
    --imdb nyud2_"$modality"_2015_"$test_set" \
    --cfg scripts/alexnet_rgb_real/config.prototxt

