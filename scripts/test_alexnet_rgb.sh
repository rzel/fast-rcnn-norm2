set -x
tr_set='trainval'
test_set='test'

modality="norm"
  PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 1 \
    --def scripts/alexnet_rgb/test.prototxt \
    --net /nfs.yoda/xiaolonw/fast_rcnn/models_norm/alexnet_rgb/fast_rcnn_iter_90000.caffemodel \
    --imdb nyud2_"$modality"_2015_"$test_set" \
    --cfg scripts/alexnet_rgb/config.prototxt
