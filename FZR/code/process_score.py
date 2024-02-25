import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from tensorboardX import SummaryWriter

data_path = '../data/'
dataname = 'Wiki'

relaM = np.load('../data/Wiki/embeddings/' + 'rela_matrix.npz')['relaM']
relaM_pt = torch.from_numpy(relaM)
relaM_norm = relaM_pt / relaM_pt.norm(dim=-1, keepdim=True)

cos_sim = torch.mm(relaM_norm, relaM_norm.t())
sorted, indices = torch.sort(cos_sim, 1, descending=True)

top_five_idx = indices[:, 1:6]

candidate_vecs = []
for i, idx_list in enumerate(top_five_idx):
    candidate_vecs.append(relaM[idx_list])
candidate_vecs = np.array(candidate_vecs)
candidate_vecs = torch.tensor(candidate_vecs)

position_vecs = relaM_pt
position_vecs = torch.unsqueeze(position_vecs, dim=1)
train_vecs = torch.cat((position_vecs, candidate_vecs), 1)

true_score = torch.load(data_path + dataname + '/' + 'score_tensor.pt')

x_data = train_vecs
y_data = torch.tensor(true_score / 5, dtype=torch.float32)

dataset = TensorDataset(x_data, y_data)

batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


relaM_dim = 300
hidden_dim = 100

class Emb(nn.Module):
    def __init__(self, relaM_dim, hidden_dim) -> None:
        super().__init__()
        self.mlp_1 = nn.Sequential(
            nn.Linear(relaM_dim, relaM_dim, bias=True),
            nn.Tanh()
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(relaM_dim * 2, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Sigmoid()
        )

    def get_new_emb(self, src_data):
        new_data = self.mlp_1(src_data)
        return new_data

    def forward(self, src_data):
        new_data = self.get_new_emb(src_data)
        repeat_size = src_data.shape[1] - 1
        present_data = new_data[:, 0:1, :].repeat(1, repeat_size, 1)

        new_input = torch.cat([present_data, new_data[:, 1:, :]], dim=-1)
        score = self.mlp_2(new_input)
        return score


model = Emb(relaM_dim, hidden_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# writer = SummaryWriter('logs/')
step = 0

for epoch in range(600):
    # for batch in range(181):
    for batch, (inputs, label) in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(inputs)
        outputs = torch.squeeze(outputs, dim=-1)

        loss = F.mse_loss(outputs, label)
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}], Step [{batch+1}/{539//batch_size + 1}], Loss: {loss.item():.4f}')
        step += 1
        # writer.add_scalar('train_loss', loss.cpu().detach().numpy(), global_step=step)

new_relaM = []
for rela in relaM_pt:
    new_relaM.append(model.get_new_emb(rela))

new_relaM = torch.stack(new_relaM, dim=0)
new_relaM = new_relaM.detach().numpy()

np.savez(data_path + dataname + '/embeddings/' + 'rela_matrix_exp.npz', relaM = new_relaM)
