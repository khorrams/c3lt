# Cycle-Consistent Counterfactuals by Latent Transformations (C3LT)
PyTorch implementation for C3LT, a novel counterfactual visual explanation method published at CVPR 2022.

>* Saeed Khorram, Li Fuxin. ["Cycle-Consistent Counterfactuals by Latent Transformations (C3LT)"](""), CVPR 2022.
 
 
### Dependencies
First install and activate a `python3.6` virtual environment:

```
$ python3.6 -m venv env
$ source env/bin/activate
```
You can update the pip and install the dependencies using:
```
(env) $ pip install --upgrade pip
(env) $ pip install -r req.txt
```

### Quick Start
For instance, to train CF latent transformations for classes `4` and `9` from the `mnist` dataset, one can simply run:
```
(env) $ python main.py --dataset mnist --cls_1 4 --cls_2 9
```

The hyperparameters for training can be directly passed as arguments when running the `main.py`. 
For the full list of arguments, please see `args.py`.
 
### Citation
If you use the implementation in your research, please consider citing our paper:

```
@inproceedings{khorram2022cycle,
  title={Cycle-Consistent Counterfactuals by Latent Transformations},
  author={Khorram, Saeed and Fuxin, Li},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
