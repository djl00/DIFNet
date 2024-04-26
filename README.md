# Real-time semantic segmentation with dual interaction fusion network

### Installation
You can refer to the official [Paddleseg](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/install_cn.md) documentation for deployment

### Dataset
You need to download the two dataset——CamVid and Cityscapes, and put the files in the `dataset` folder with following structure.
```
├── camvid
|    ├── train
|    ├── test
|    └── val
├── cityscapes
|    ├── gtFine
|    └── leftImg8bit           
```

### Training

- You can refer to the official Paddleseg documentation for training.[train](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/train/train_cn.md)
- If you want to use multi-card training, you need to specify the environment variable CUDA_VISIBLE_DEVICES as multi-card (if not specified, all GPUs will be used by default), and use paddle.distributed.launch to start the training script (Can not use multi-card training under Windows, because it doesn't support nccl):
```
export CUDA_VISIBLE_DEVICES=0,1,2,3 # Set 4 usable cards
python -m paddle.distributed.launch tools/train.py \
       --config configs/difnet/difnet_stdc1_cityscapes_1024x512_scale0.75_160k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

### Evaluation
- After the training, the user can use the evaluation script val.py to evaluate the effect of the model..
```
python tools/val.py \
       --config configs/difnet/difnet_stdc1_cityscapes_1024x512_scale0.75_160k.yml \
       --model_path output/iter_160000/model.pdparams
```
### Predict
- For those dataset that do not provide label on the test set (e.g. Cityscapes), you can use `predict.py` to save all the output images, then submit to official webpage for evaluation.
```
python tools/predict.py \
       --config configs/difnet/difnet_stdc1_cityscapes_1024x512_scale0.75_160k.yml \
       --model_path output/iter_160000/model.pdparams \
       --image_path data/cityscapes/leftImg8bit/test \
       --save_dir output/result
```


### Inference Speed
- You can refer to the official [Paddleseg documentation](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/deployment/inference/python_inference_cn.md) for deploying inference time.
