# Train the HHA Alexnet model from the supervision transfer initialization weights
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/train_net.py --gpu 1 \
  --weights /nfs.yoda/xiaolonw/fast_rcnn/fast-rcnn-distillation/data/init_models/CaffeNet/CaffeNet.v2.caffemodel \
  --solver scripts/alexnet_rgb/solver.prototxt \
  --imdb nyud2_norm_2015_trainval \
  --cfg scripts/alexnet_rgb/config.prototxt \
  --iters 100000 \
  2>&1 | tee scripts/alexnet_rgb/train.log

