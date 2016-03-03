# Testing alexnet_rgb and alexnet_hha models
model='alexnet_rgb_alexnet_hha'; tr_set='trainval'; test_set='test'; modality="images+hha";
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 0 \
  --def output/training_demo/$model/test.prototxt.$modality \
  --net output/training_demo/$model/nyud2_images+hha_2015_$tr_set/fast_rcnn_iter_40000.caffemodel \
  --imdb nyud2_"$modality"_2015_"$test_set" \
  --cfg output/training_demo/$model/config.prototxt."$modality"