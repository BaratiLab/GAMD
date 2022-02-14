# Code for GAMD

To train the model with the generated data simply run one of the ```train_network_x.py``` scripts in the corresponding folder (LJ, Water).

For example, to train model on Lennard Jones data using paper's configuration.
```bash
cd LJ
python train_network_lj.py \
--cp_dir model_ckpt_lj \   # where to store checkpoints
--min_epoch 30 \
--max_epoch 30 \
--batch_size 1 \
--encoding_size 128 \      # dimension of node embedding
--hidden_dim 128 \         # dimension of hidden units
--edge_embedding_dim 128 \ # dimension of edge embedding
--loss mae \               # loss function: mean absolute error
--data_dir $PATH_TO_LJ_DATA \  # replace $PATH_TO_LJ_DATA with the directory data folder located at
--use_layer_norm
```

To train on DFT data, it is recommended to use a larger model.
```bash
cd water
python train_network_real_large.py \
--cp_dir model_ckpt_dft \   # where to store checkpoints
--min_epoch 800 \
--max_epoch 800 \
--batch_size 8 \
--encoding_size 256 \      # dimension of node embedding
--hidden_dim 128 \         # dimension of hidden units
--edge_embedding_dim 256 \ # dimension of edge embedding
--conv_layer 5 \           # number of message passing layers
--loss mae \               # loss function: mean absolute error
--data_dir $PATH_TO_DFT_DATA \  # replace $PATH_TO_DFT_DATA with the directory data folder located at
--use_layer_norm
```

The pretrained model checkpoints are at corresponding subfolders. LJ's checkpoint is at```LJ/model_ckpt_lj```, water's checkpoint is at```water/model_ckpt_x```.

To run the simulation using GAMD, please refer to the test_langevin.py/test_nosehoover.py scripts in the subfolder ```LJ/test_script/``` and ```water/test_script/```. As DGL does not support Torchscript currently, the integration of GAMD into OpenMM is achieved in a way with a lot of overheads that requires copying data from the simulation context and then update them.
