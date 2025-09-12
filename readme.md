# When GNNs meet symmetry in ILPs: an orbit-based feature augmentation approach

This repository contains the code for the NeurIPS 2024 paper: **[When GNNs meet symmetry in ILPs: an orbit-based feature augmentation approach](https://arxiv.org/pdf/2501.14211)**.

## Environment Setup
To run this code, you need the following dependencies:
- Python 3.9.19
- pyg 2.5.3
- pytorch 2.4.0
- pyscipopt 3.5.0



## Data preparation

Follow instructions [here](./data/README.md) to prepare the data.

## Training the Model

To train the model, you can use the following bash commands:

```bash
epoch=100
sampleTimes=8
for dataset in BIP BBP SMSP
do
    python train.py --Aug empty --dataset $dataset  --epoch $epoch --sampleTimes $sampleTimes
    python train.py --Aug uniform --dataset $dataset  --epoch $epoch --sampleTimes $sampleTimes
    python train.py --Aug pos --dataset $dataset  --epoch $epoch --sampleTimes $sampleTimes
    python train.py --Aug orbit --dataset $dataset  --epoch $epoch --sampleTimes $sampleTimes
    python train.py --Aug group --dataset $dataset  --epoch $epoch --sampleTimes $sampleTimes
done
```
## Evaluation

### Draw loss curves

After training, the validation curves of different methods can be drawn by running Matlab script

```plaintext
draw_loss.m
```


### Get Top-m% error

statistics regarding Top-m% error can be calculated by running
```
python read_top_m_error.py
```

the results will be reported in `./handisTable_valid.xlsx`
