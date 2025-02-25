import torch
import torch.nn as nn

class RvNN(nn.Module):
	def __init__(self, args, input_dim):
		super().__init__()
		self.args = args

		self.E = nn.Linear(input_dim, args.hidden_dim, bias=False)
		self.Wr = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
		self.Wz = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
		self.Ur = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
		self.Uz = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
		self.Wh = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
		self.Uh = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
		self.decoder = nn.Linear(args.hidden_dim, args.n_class)

	def forward_step(self, x_hat, h, j_level, adj, level):
		cur_level = (level == j_level).float().unsqueeze(1)
		level_adj = adj * cur_level
		hs = level_adj.mm(h)
		rj = torch.sigmoid(self.Wr(x_hat) + self.Ur(hs))
		zj = torch.sigmoid(self.Wz(x_hat) + self.Uz(hs))
		hj_hat = torch.tanh(self.Wh(x_hat) + self.Uh(hs*rj))
		hj = (1. - zj) * hs + zj * hj_hat
		h = (cur_level == 0).float() * h + cur_level * hj
		return h

class BURvNN(RvNN):
	def __init__(self, args, input_dim):
		super().__init__(args, input_dim)

	def forward(self, x):
		adj = x['adj']
		tfidf = x['tfidf']
		level = x['level']

		seq_size = adj.size(0)

		x_hat = self.E(tfidf)
		h = torch.zeros((seq_size, self.args.hidden_dim), device=self.args.device)

		max_level = int(level.max().cpu().item())
		for j in range(max_level, -1, -1):
			h = self.forward_step(x_hat, h, j, adj, level)

		root_idx = (level == 0).float().unsqueeze(0)
		root_rep = root_idx.mm(h)
		logit = self.decoder(root_rep)
		return logit

class TDRvNN(RvNN):
	def __init__(self, args, input_dim):
		super().__init__(args, input_dim)

	def forward(self, x):
		adj = x['adj'].permute(1, 0)
		tfidf = x['tfidf']
		level = x['level']

		seq_size = adj.size(0)

		x_hat = self.E(tfidf)
		h = torch.zeros((seq_size, self.args.hidden_dim), device=self.args.device)

		max_level = int(level.max().cpu().item())
		# for j in range(max_level, -1, -1):
		for j in range(max_level+1):
			h = self.forward_step(x_hat, h, j, adj, level)

		leaf_mask = (level == max_level).float().unsqueeze(1).repeat(1, h.size(1))
		zero_vec = -9e15 * torch.ones_like(h)
		h = torch.where(leaf_mask==0, h, zero_vec)
		root_rep = torch.max(h, dim=0, keepdim=True)[0]
		logit = self.decoder(root_rep)
		return logit
