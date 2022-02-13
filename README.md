# GAMD
Data and code for graph neural network accelerated molecular dynamics

## Dependencies

- Pytorch 1.7.1
- jax-md (latest), jax and jaxlib 0.1.67 (for gpu support)
- Pytorch-lightning 1.3.0
- DGL 0.7.0 (https://www.dgl.ai/)
- Cupy 9.1.0
- OpenMM (https://openmm.org/) & OpenMMTools (https://openmmtools.readthedocs.io/en/0.18.1/)

## Data generation

The data generation scripts using classical MD are in the ```dataset``` subfolder.
For example, to generate the data for Lennard Jones particles:
```bash
cd dataset
python generate_lj_data.py
```
    
The generated data can also be downloaded from below links: </br>
[Lennard Jones](https://drive.google.com/file/d/1jJdTAnhps1EIHDaBfb893fruaLPJzYKI/view?usp=sharing)  
[TIP3P](https://drive.google.com/file/d/18uvKVtN8Xm_5w7AJuezFiR1d2AqlHFKn/view?usp=sharing)  
[TIP4P-Ew](https://drive.google.com/file/d/1jBk78hN4ZPC9x-YXnznUzxFnXnpeKFRi/view?usp=sharing)

The DFT data used in this work is derived from the paper:
How van der Waals interactions determine the unique properties of water [(pdf)](https://www.pnas.org/content/113/30/8368.short) 
[(data link)](https://zenodo.org/record/2634098#.Ygl0QnVKg5k). </br>

We provide our processed data (.npz fromat) at [data link](https://drive.google.com/file/d/1b9P7EvIliGupN9ZIJpMGZzkm4ttG9Ul6/view?usp=sharing).
    
## Training

The training scripts and pretrained models are in the ```code``` subfolder. Please refer to the subfolder for more details.
