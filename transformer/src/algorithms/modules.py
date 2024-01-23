import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import algorithms.vit as vit
import utils
import math


def _get_out_shape_cuda(in_shape, layers):
	x = torch.randn(*in_shape).cuda().unsqueeze(0)
	return layers(x).squeeze(0).shape

def _get_out_shape(in_shape, layers):
	x = torch.randn(*in_shape).unsqueeze(0)
	return layers(x).squeeze(0).shape


def gaussian_logprob(noise, log_std):
	"""Compute Gaussian log probability."""
	residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
	return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
	"""Apply squashing function.
	See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
	"""
	mu = torch.tanh(mu)
	if pi is not None:
		pi = torch.tanh(pi)
	if log_pi is not None:
		log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
	return mu, pi, log_pi


def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		# delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class CenterCrop(nn.Module):
	def __init__(self, size):
		super().__init__()
		assert size in {84, 96, 100, 112}, f'unexpected size: {size}'
		self.size = size

	def forward(self, x):
		assert x.ndim == 4, 'input must be a 4D tensor'
		if x.size(2) == self.size and x.size(3) == self.size:
			return x
		assert x.size(3) == 100, f'unexpected size: {x.size(3)}'
		if self.size == 96:
			p = 2
		elif self.size == 84:
			p = 8
		return x[:, :, p:-p, p:-p]


class Time2Space(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		x = x.view(x.size(0), x.size(1)//3, 3, x.size(-2), x.size(-1)) # (B, T, C, H, W)
		return torch.cat(torch.unbind(x, dim=1), dim=-1) # (B, C, H, (TW))


class NormalizeImg(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x/255.


class Flatten(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)


# class RLProjection(nn.Module):
# 	def __init__(self, in_shape, out_dim):
# 		super().__init__()
# 		self.out_dim = out_dim
# 		self.projection = nn.Sequential(
# 			nn.Linear(in_shape[0], out_dim),
# 			nn.LayerNorm(out_dim),
# 			nn.Tanh()
# 		)
# 		self.apply(weight_init)
	
# 	def forward(self, x):
# 		return self.projection(x)



class RLProjection(nn.Module):
	def __init__(self, in_shape, out_dim, patch_num):
		super().__init__()
		self.out_dim = out_dim
		self.projection = nn.Sequential(
			nn.Linear(int(in_shape[0]/(patch_num+1)), out_dim),
			nn.LayerNorm(out_dim),
			nn.Tanh()
		)
		self.apply(weight_init)
	
	def forward(self, x):
		return self.projection(x)



class SODAMLP(nn.Module):
	def __init__(self, projection_dim, hidden_dim, out_dim):
		super().__init__()
		self.out_dim = out_dim
		self.mlp = nn.Sequential(
			nn.Linear(projection_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, out_dim)
		)
		self.apply(weight_init)

	def forward(self, x):
		return self.mlp(x)


class SharedTransformer(nn.Module):
	def __init__(self, obs_shape, patch_size=8, embed_dim=128, depth=4, num_heads=8, mlp_ratio=1., qvk_bias=False):
		super().__init__()
		assert len(obs_shape) == 3
		self.frame_stack = obs_shape[0]//3
		self.img_size = obs_shape[-1]
		self.patch_size = patch_size
		self.embed_dim = embed_dim
		self.depth = depth
		self.num_heads = num_heads
		self.mlp_ratio = mlp_ratio
		self.qvk_bias = qvk_bias

		self.preprocess = nn.Sequential(CenterCrop(size=self.img_size), NormalizeImg())
		self.transformer = vit.VisionTransformer(
			img_size=self.img_size,
			patch_size=patch_size,
			in_chans=self.frame_stack*3,
			embed_dim=embed_dim,
			depth=depth,
			num_heads=num_heads,
			mlp_ratio=mlp_ratio,
			qkv_bias=qvk_bias,
		).cuda()
		self.out_shape = _get_out_shape_cuda(obs_shape, nn.Sequential(self.preprocess, self.transformer))

	def forward(self, x):
		x = self.preprocess(x)
		return self.transformer(x)


class SharedCNN(nn.Module):
	def __init__(self, obs_shape, num_layers=11, num_filters=32):
		super().__init__()
		assert len(obs_shape) == 3
		self.img_size = obs_shape[-1]
		self.num_layers = num_layers
		self.num_filters = num_filters

		self.layers = [CenterCrop(size=self.img_size), NormalizeImg()]
		self.layers.append(nn.Conv2d(obs_shape[0], num_filters, 3, stride=2))
		for i in range(1, num_layers):
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(obs_shape, self.layers)
		self.apply(weight_init)

	def forward(self, x):
		return self.layers(x)


class HeadCNN(nn.Module):
	def __init__(self, in_shape, num_layers=0, num_filters=32):
		super().__init__()
		self.layers = []
		for _ in range(0, num_layers):
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers.append(Flatten())
		self.layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(in_shape, self.layers)
		self.apply(weight_init)

	def forward(self, x):
		return self.layers(x)


class Encoder(nn.Module):
	def __init__(self, shared, head, projection):
		super().__init__()
		self.shared = shared
		self.head = head
		self.projection = projection
		self.out_dim = projection.out_dim

	def forward(self, x, detach=False):
		x = self.shared(x)[:, 0]
		x = self.head(x)
		if detach:
			x = x.detach()

		return self.projection(x)


class Actor(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim, log_std_min, log_std_max):
		super().__init__()
		self.encoder = encoder
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		self.mlp = nn.Sequential(
			nn.Linear(self.encoder.out_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, 2 * action_shape[0])
		)
		self.mlp.apply(weight_init)

	def forward(self, x, compute_pi=True, compute_log_pi=True, detach=False):
		x = self.encoder(x, detach)
		mu, log_std = self.mlp(x).chunk(2, dim=-1)
		log_std = torch.tanh(log_std)
		log_std = self.log_std_min + 0.5 * (
			self.log_std_max - self.log_std_min
		) * (log_std + 1)

		if compute_pi:
			std = log_std.exp()
			noise = torch.randn_like(mu)
			pi = mu + noise * std
		else:
			pi = None
			entropy = None

		if compute_log_pi:
			log_pi = gaussian_logprob(noise, log_std)
		else:
			log_pi = None

		mu, pi, log_pi = squash(mu, pi, log_pi)

		return mu, pi, log_pi, log_std


class QFunction(nn.Module):
	def __init__(self, obs_dim, action_dim, hidden_dim):
		super().__init__()
		self.trunk = nn.Sequential(
			nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)
		self.apply(weight_init)

	def forward(self, obs, action):
		assert obs.size(0) == action.size(0)
		return self.trunk(torch.cat([obs, action], dim=1))


class Critic(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.Q1 = QFunction(
			self.encoder.out_dim, action_shape[0], hidden_dim
		)
		self.Q2 = QFunction(
			self.encoder.out_dim, action_shape[0], hidden_dim
		)

	def forward(self, x, action, detach=False):
		x = self.encoder(x, detach)
		return self.Q1(x, action), self.Q2(x, action)


class CURLHead(nn.Module):
	def __init__(self, encoder):
		super().__init__()
		self.encoder = encoder
		self.W = nn.Parameter(torch.rand(encoder.out_dim, encoder.out_dim))

	def compute_logits(self, z_a, z_pos):
		"""
		Uses logits trick for CURL:
		- compute (B,B) matrix z_a (W z_pos.T)
		- positives are all diagonal elements
		- negatives are all other elements
		- to compute loss use multiclass cross entropy with identity matrix for labels
		"""
		Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
		logits = torch.matmul(z_a, Wz)  # (B,B)
		logits = logits - torch.max(logits, 1)[0][:, None]
		return logits


class InverseDynamics(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.mlp = nn.Sequential(
			nn.Linear(2*encoder.out_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, action_shape[0])
		)
		self.apply(weight_init)

	def forward(self, x, x_next):
		h = self.encoder(x)
		h_next = self.encoder(x_next)
		joint_h = torch.cat([h, h_next], dim=1)
		return self.mlp(joint_h)


class SODAPredictor(nn.Module):
	def __init__(self, encoder, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.mlp = SODAMLP(
			encoder.out_dim, hidden_dim, encoder.out_dim
		)
		self.apply(weight_init)

	def forward(self, x):
		return self.mlp(self.encoder(x))



class Decoder(torch.nn.Module):
	def __init__(self,
				 encoder_for_aux,
				 embedding_dim,
				 n_heads=8,
				 tanh_clipping=10.0,
				 ):
		super(Decoder, self).__init__()
		self.embedding_dim = embedding_dim
		self.n_heads = n_heads
		self.tanh_clipping = tanh_clipping
		self.encoder_for_aux=encoder_for_aux

		self.Wq_graph = torch.nn.Linear(embedding_dim,
										embedding_dim,
										bias=False)
		self.Wq_first = torch.nn.Linear(embedding_dim,
										embedding_dim,
										bias=False)
		self.Wq_last = torch.nn.Linear(embedding_dim,
									   embedding_dim,
									   bias=False)
		self.v1 = nn.Parameter(torch.Tensor(embedding_dim))
		self.v2 = nn.Parameter(torch.Tensor(embedding_dim))
		self.v1.data.uniform_(-1, 1)
		self.v2.data.uniform_(-1, 1)

		self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
		self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
		self.logit_Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)

		self.multi_head_combine = torch.nn.Linear(embedding_dim, embedding_dim)

		self.q_graph = None  # saved q_fixed(graph)--cls, for multi-head attention
		self.q_first = None  # saved q_first_node, for multi-head attention
		self.glimpse_k = None  # saved key, for multi-head attention
		self.glimpse_v = None  # saved value, for multi-head_attention
		self.logit_k = None  # saved, for single-head attention
		self.group_ninf_mask = None

	def reset(self, fixed_content_cls, patch_embeddings):
		# fixed_content_cls.shape = [B, 1, embedding_dim]
		# patch_embeddings.shape = [B, N, embedding_dim]
		self.batch_index_vector = torch.arange(patch_embeddings.size(0))
		self.patch_embeddings = patch_embeddings
		self.fixed_content_cls = fixed_content_cls
		self.q_graph = utils.make_heads(self.Wq_graph(self.fixed_content_cls),
										self.n_heads)
		self.q_first = None
		self.glimpse_k = utils.make_heads(self.Wk(self.patch_embeddings), self.n_heads)
		self.glimpse_v = utils.make_heads(self.Wv(self.patch_embeddings), self.n_heads)
		self.logit_k = utils.make_heads(self.logit_Wk(self.patch_embeddings), 1)
		self.group_ninf_mask = torch.zeros(*self.patch_embeddings.size()[:-1]).cuda()


	def forward(self, last_patch=None):
		# return [B,N]
		# input: index for last selected patches [B],index begin from 0,end in N-1
		B, N, H = self.patch_embeddings.shape

		if last_patch is None:
			placeholder_v1 = utils.make_heads(self.Wq_first(self.v1[None, None, :].expand(B, 1, -1)), self.n_heads)
			placeholder_v2 = utils.make_heads(self.Wq_last(self.v2[None, None, :].expand(B, 1, -1)), self.n_heads)
			glimpse_q = self.q_graph + placeholder_v1 + placeholder_v2
		else:
			self.group_ninf_mask[self.batch_index_vector, last_patch] = -np.inf
			last_patch_index = last_patch.view(B, 1, 1).expand(-1, -1, H)
			last_patch_embedding = self.patch_embeddings.gather(1, last_patch_index)
			if self.q_first is None:
				self.q_first = utils.make_heads(self.Wq_first(last_patch_embedding),
												self.n_heads)
			q_last = utils.make_heads(self.Wq_last(last_patch_embedding),
									  self.n_heads)
			glimpse_q = self.q_graph + self.q_first + q_last

		attn_out = utils.multi_head_attention(q=glimpse_q,
											  k=self.glimpse_k,
											  v=self.glimpse_v,
											  mask=self.group_ninf_mask)

		final_q = self.multi_head_combine(attn_out)
		score = torch.matmul(final_q, self.logit_k.squeeze(1).transpose(1, 2)) / math.sqrt(H)
		score_clipped = self.tanh_clipping * torch.tanh(score)
		score_masked = score_clipped.squeeze() + self.group_ninf_mask

		probs = F.softmax(score_masked, dim=1)
		assert (probs == probs).all(), "Probs should not contain any nans!"
		return probs # [B,N]
		# assert (score_masked == score_masked).all(), "score_masked should not contain any nans!"
		# return score_masked


class Encoder_for_aux(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def forward(self, x):
        x = self.shared(x)
        return x