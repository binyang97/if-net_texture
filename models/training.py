from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from glob import glob
import numpy as np




class Trainer(object):

    def __init__(self, model, device, train_dataset, val_dataset, exp_name, optimizer='Adam'):
        self.model = model.to(device)
        self.device = device
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters())
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format( exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.val_min = None


    def train_step(self,batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_loss(self,batch):
        device = self.device

        p = batch.get('grid_coords').to(device)
        gt_rgb = batch.get('rgb').to(device)
        inputs = batch.get('inputs').to(device)

        # print(gt_rgb)
        # p = p.unsqueeze(1).unsqueeze(1)
        # print(F.grid_sample(inputs, p, padding_mode='border'))
        # General points

        # print(p[:,:3])
        pred_rgb = self.model(p,inputs)
        pred_rgb =  pred_rgb.transpose(-1,-2)
        # print(gt_rgb.shape)
        loss_i = torch.nn.L1Loss(reduction='none')(pred_rgb, gt_rgb)  # out = (B,num_points,3)

        loss = loss_i.sum(-1).mean() # loss_i summed 3 rgb channels for all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)

        return loss

    def train_model(self, epochs):
        loss = 0
        start = self.load_checkpoint()

        for epoch in range(start, epochs):
            sum_loss = 0
            print('Start epoch {}'.format(epoch))
            train_data_loader = self.train_dataset.get_loader()

            if epoch % 1 == 0:
                self.save_checkpoint(epoch)
                val_loss = self.compute_val_loss()

                if self.val_min is None:
                    self.val_min = val_loss

                if val_loss < self.val_min:
                    self.val_min = val_loss
                    # for path in glob(self.exp_path + 'val_min=*'):
                    #     os.remove(path)
                    np.save(self.exp_path + 'val_min',[epoch,val_loss])


            for batch in train_data_loader:
                loss = self.train_step(batch)
                print("Current loss: {}".format(loss))
                sum_loss += loss




    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):
            torch.save({'epoch':epoch,'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch

    def compute_val_loss(self):
        self.model.eval()

        sum_val_loss = 0
        num_batches = 15
        for _ in range(num_batches):
            try:
                val_batch = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_dataset.get_loader().__iter__()
                val_batch = self.val_data_iterator.next()

            sum_val_loss += self.compute_loss( val_batch).item()

        return sum_val_loss / num_batches