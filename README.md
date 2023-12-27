# SegFormer
SegFormer

## Dependecy installation
```python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

## Dataset
Download dataset [ADE20k](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) and upload it into directory: `./data/ADEChallengeData2016`

## Train
```python
python train.py --config './configs/segformer.yaml' --train_url './experiments/ckpts/tensorboard'
```
`--config`: configuration file (training, dataset, model)  
`--train_url`: directory to save tensoboard outputs

or:
```python
python run.py
```

Multi-GPU training is support, the number of available GPUs should be change in cofiguration file: `./configs/segformer.yaml`

## Inference:
```python
python inference.py \
  --config './configs/segformer.yaml' \
  --weight './experiments/pretrained/pytorch_model.bin' \
  --input_path './data/ADEChallengeData2016/images/training' \
  --output_path "./output" \
  --image_size 512 \
  --ext 'jpg'
```


## Docker
Dockerfile: Dockerfile
- [ ] Test Dockerfile

```python
docker compose up
```

## Logging
Launch tensorboard after training:
```python
tensorboard --logdir experiments/tensorboard
```

1. Leaning rate
2. Metrics (pixel-wise accuracy)
3. Loss
4. Image (from left to right: image, gt_mask, pred_mask)

![image](https://github.com/Xrenya/SegFormer/assets/51479797/08460bef-08d8-4765-baab-27373c262eb8)

## Test
Run pytest:
```python
pytest .
```

## Code Formatting
Sorting imports:
```python
isort *
```

Code formatter:
```python
yapf -i *python file*
```

Checks codebase for errors, styling issues and complexity:
```python
flake8 *python file*
```

## Test Env
The code was tested in Linux enviroment using 1 GPU.
The training enviroment does support multi-gpu training (8 GPUs), but was not test.
