import os
import torch
import torch.nn as nn
from torch.optim import Adam
from core.models import action_cond_modernn, action_cond_predrnn, crevnet, modernn


class Model(object):
    def __init__(self, configs, parser=None):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        self.parser = parser
        networks_map = {
            'modernn' : modernn.RNN,
            'crevnet' : crevnet.RNN,
            'action_cond_predrnn' : action_cond_predrnn.RNN,
            'action_cond_modernn' : action_cond_modernn.RNN, 

        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)



        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.MSE_criterion = nn.MSELoss()
        self.alpha = 0.9

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])


    def train(self, frames, mask, iter):
        #print(frames.shape)
        frames_tensor = torch.FloatTensor(frames.copy()).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()

        next_frames, loss = self.network(frames_tensor, mask_tensor, True)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()


    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames.copy()).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        with torch.no_grad():
            next_frames,  _ = self.network(frames_tensor, mask_tensor, False)
        return next_frames.detach().cpu().numpy()
