from monai.networks.nets import SegResNet #SwinUNETR #VNet #UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss
from torch.optim.lr_scheduler import ExponentialLR
import torch
from preprocess import prepare
from utilities import train

#from monai.inferers import sliding_window_inference #SwinUNet

data_dir = '/content/drive/MyDrive/Brain-Tumor/Data_Train_Test'
model_dir = '/content/drive/MyDrive/Brain-Tumor/results' 
data_in = prepare(data_dir, cache=True)

device = torch.device("cuda:0")

#model = UNet(
 #   spatial_dims=3,
 #   in_channels=1,
 #   out_channels=2,
 #   channels=(16, 32, 64, 128, 256), 
 #   strides=(2, 2, 2, 2),
 #   num_res_units=2,
 #   norm=Norm.BATCH,
 #   dropout=0.25 #dropout rate 
#).to(device)


model = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    init_filters=16,
    blocks_down=[1, 2, 2, 4],
    upsample_mode="deconv",
    dropout_prob=0.25
).to(device)

"""
model = VNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    act=('elu', {'inplace': True}),
    dropout_prob_down=0.4,
    dropout_prob_up=(0.4, 0.4),
    dropout_dim=3,
    bias=False,
).to(device)
"""
"""
model = SwinUNETR(
    in_channels=1,
    out_channels=2,
    feature_size=48,
    drop_rate=0.2,
    attn_drop_rate=0.2,
    dropout_path_rate=0.2,
    use_checkpoint=True,
).to(device)
"""

#loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True).to(device)
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.AdamW(model.parameters(), 1e-4, weight_decay=1e-5, amsgrad=True)
scheduler = ExponentialLR(optimizer, gamma=0.995)

if __name__ == '__main__':
    train(model, data_in, loss_function, optimizer, 200, model_dir, scheduler=scheduler)