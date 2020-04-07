import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prf_fn
from sklearn.metrics import accuracy_score
from rvnn import BURvNN, TDRvNN

class Wrapper():
	def __init__(self, args, loader):
		self.args = args
		self.loader, self.fold = loader
		self.train_loader, self.test_loader, self.vocab_size = self.loader.generate_fold_data(self.fold)

		if args.model == 'BU_RvNN':
			self.model = BURvNN(args, self.vocab_size)
		elif args.model == 'TD_RvNN':
			self.model = TDRvNN(args, self.vocab_size)
		else:
			raise NotImplementedError

		self.model.to(args.device)
		self.parameters = [p for p in self.model.parameters() if p.requires_grad]
		self.optimizer = optim.Adam(self.parameters)
		# self.optimizer = optim.SGD(self.parameters, lr=0.005, momentum=0.9)
		self.loss_fn = nn.CrossEntropyLoss()

	def run(self):
		best_prf = None
		for epoch in range(self.args.n_epoch):
			self.train(epoch)
			prf = self.evaluate()
			# if best_prf is None or prf[-1][-1] > best_prf[-1][-1]:
			print('------------Epoch %d--------------'%epoch)
			for key in range(len(self.args.label_mapping)):
				print('Label %s\tP %0.3f\tR %0.3f\tF %0.3f' % (self.args.label_mapping[key], *prf[key]))
			print('Accuracy\t\t\t%0.3f' % prf[-2])
			print('Macro\t\t\tP %0.3f\tR %0.3f\tF %0.3f' % prf[-1])
			if best_prf is None or prf[-2] > best_prf[-2]:
				best_prf = prf
				print('Accuracy Improved to \t\t\t%0.3f' % prf[-2])
		return best_prf

	def train(self, epoch):
		train_adj, train_tfidf, train_level, train_label = self.train_loader
		train_key = list(sorted(train_adj.keys()))
		np.random.shuffle(train_key)
		with tqdm(total=len(train_key)) as pbar:
			for key in train_key:
				x = {'adj': train_adj[key], 'tfidf': train_tfidf[key], 'level': train_level[key]}
				y_true = train_label[key]
				loss = self.update(x, y_true)
				pbar.set_description("Loss %0.4f" % (loss))
				pbar.update()

	def update(self, x, y_true):
		self.model.train()
		y_pred = self.model(x)
		loss = self.loss_fn(y_pred, y_true)

		self.model.zero_grad()
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item()

	def evaluate(self):
		y_pair = [[], []]
		y_emotion = [[], []]
		y_cause = [[], []]

		all_y_true = []
		all_y_pred = []
		test_adj, test_tfidf, test_level, test_label = self.test_loader
		test_key = list(sorted(test_adj.keys()))
		with torch.no_grad():
			for key in test_key:
				self.model.eval()
				x = {'adj': test_adj[key], 'tfidf': test_tfidf[key], 'level': test_level[key]}
				y_true = test_label[key].squeeze()
				y_pred = self.model(x).squeeze()
				all_y_true.append(y_true.cpu().item())
				all_y_pred.append(y_pred.topk(1)[1].cpu().item())
		all_prf = prf_fn(all_y_true, all_y_pred, average=None)
		all_prf = [(p, r, f) for p, r, f in zip(*all_prf[:-1])]
		macro_prf = prf_fn(all_y_true, all_y_pred, average='macro')[:-1]
		accuracy = accuracy_score(all_y_true, all_y_pred)
		all_prf = all_prf + [accuracy, macro_prf]
		return all_prf
