# CSQ_pytorch

This repository is re-implementation of CSQ(Central Similarity Quantization) with MFNet(Multi-Fiber Net)
<br/><br/><br/><br/>
## Installation

1. **Get the code.** 
```
  git clone https://github.com/Jo-won/CSQ_pytorch.git
```
2. **Check requirement.txt**

3. **Go to Wandb and create an account.**
<br/><br/><br/><br/>

## Dataset  
  
Running this code requires UCF101 dataset. (available [here](https://www.crcv.ucf.edu/data/UCF101.php))  
The data path can be checked below.  
Please move ```trainlist01_ilp.txt``` and ```testlist01_ilp.txt``` inside ```ucf101_txt folder```

``` 
<DATA_PATH>

+-- DATA_ROOT
|   +-- video
|   |   +-- ApplyEyeMakeup
|   |   |   +-- v_ApplyEyeMakeup_g01_c01.avi
|   |   |   +-- ...
|   |   +-- ApplyLipstick
|   |   +-- ...
|   +-- ucfTrainTestlist
|   |   +-- trainlist01_ilp.txt
|   |   +-- testlist01_ilp.txt

```
<br/><br/><br/><br/>

## Usage
- **Train & Vaild**  
Please set the ```model_root``` folder to be saved at ```opt.py```,  
set the ```data_root``` in the ```train_ucf101_base.sh``` file,  
and then use the command below.  
Validation proceeds with the same videos as the test.
```
bash train_ucf101_base.sh
```
<br/><br/><br/><br/>
- **Result**   
When CSQ 64bits,

| mAP@100 (CSQ 64bits) | Original Paper | Our re-implementation |
|:--------------------:|:--------------:|:---------------------:|
|        UCF101        |      0.874     |       **0.934**       |

![image](https://user-images.githubusercontent.com/46413594/115952832-47aeab80-a523-11eb-997f-03927a4aec7b.png)
<br/><br/><br/><br/>

## Reference
- CSQ official code : https://github.com/yuanli2333/Hadamard-Matrix-for-hashing
- MFNet official code : https://github.com/cypw/PyTorch-MFNet
