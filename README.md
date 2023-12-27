# SegFormer
SegFormer


## Logging
Launch tensorboard after training:
```python
tensorboard --logdir experiments/tensorboard
```

1. Leaning rate
2. Metrics (pixel-wise accuracy)
3. Loss
4. Image

![image](https://github.com/Xrenya/SegFormer/assets/51479797/08460bef-08d8-4765-baab-27373c262eb8)

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
