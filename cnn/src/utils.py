import torch
import numpy as np
import os
import glob
import json
import random
import augmentations
import subprocess
from datetime import datetime
import math
import torch.nn.functional as F
import time
from skimage.util.shape import view_as_windows


class eval_mode(object):
	def __init__(self, *models):
		self.models = models

	def __enter__(self):
		self.prev_states = []
		for model in self.models:
			self.prev_states.append(model.training)
			model.train(False)

	def __exit__(self, *args):
		for model, state in zip(self.models, self.prev_states):
			model.train(state)
		return False


def soft_update_params(net, target_net, tau):
	for param, target_param in zip(net.parameters(), target_net.parameters()):
		target_param.data.copy_(
			tau * param.data + (1 - tau) * target_param.data
		)


def cat(x, y, axis=0):
	return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


def write_info(args, fp):
	data = {
		'timestamp': str(datetime.now()),
		'git': subprocess.check_output(["git", "describe", "--always"]).strip().decode(),
		'args': vars(args)
	}
	with open(fp, 'w') as f:
		json.dump(data, f, indent=4, separators=(',', ': '))


def load_config(key=None):
	path = os.path.join('setup', 'config.cfg')
	with open(path) as f:
		data = json.load(f)
	if key is not None:
		return data[key]
	return data


def make_dir(dir_path):
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def listdir(dir_path, filetype='jpg', sort=True):
	fpath = os.path.join(dir_path, f'*.{filetype}')
	fpaths = glob.glob(fpath, recursive=True)
	if sort:
		return sorted(fpaths)
	return fpaths


def prefill_memory(obses, capacity, obs_shape):
	"""Reserves memory for replay buffer"""
	c,h,w = obs_shape
	for _ in range(capacity):
		frame = np.ones((3,h,w), dtype=np.uint8)
		obses.append(frame)
	return obses


class ReplayBuffer(object):
	"""Buffer to store environment transitions"""
	def __init__(self, obs_shape, action_shape, capacity, batch_size, aux_batch_size,image_size=72, prefill=True):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.device=device
		self.capacity = capacity
		self.batch_size = batch_size
		self.aux_batch_size=aux_batch_size
		self.image_size = image_size

		self._obses = []
		if prefill:
			self._obses = prefill_memory(self._obses, capacity, obs_shape)
		self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
		self.rewards = np.empty((capacity, 1), dtype=np.float32)
		self.not_dones = np.empty((capacity, 1), dtype=np.float32)

		self.idx = 0
		self.full = False

	def add(self, obs, action, reward, next_obs, done):
		obses = (obs, next_obs)
		if self.idx >= len(self._obses):
			self._obses.append(obses)
		else:
			self._obses[self.idx] = (obses)
		np.copyto(self.actions[self.idx], action)
		np.copyto(self.rewards[self.idx], reward)
		np.copyto(self.not_dones[self.idx], not done)

		self.idx = (self.idx + 1) % self.capacity
		self.full = self.full or self.idx == 0

	def _get_idxs(self, n=None):
		if n is None:
			n = self.batch_size
		return np.random.randint(
			0, self.capacity if self.full else self.idx, size=n
		)
	
	def _get_aux_idxs(self,n=None):
		if n is None:
			n=self.aux_batch_size
		return np.random.randint(
			0,self.capacity if self.full else self.idx, size=n
		)

	def _encode_obses(self, idxs):
		obses, next_obses = [], []
		for i in idxs:
			obs, next_obs = self._obses[i]
			obses.append(np.array(obs, copy=False))
			next_obses.append(np.array(next_obs, copy=False))
		return np.array(obses), np.array(next_obses)

	def sample_soda(self, n=None):
		idxs = self._get_idxs(n)
		obs, _ = self._encode_obses(idxs)
		return torch.as_tensor(obs).cuda().float()

	def sample_curl(self, n=None):
		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		pos = augmentations.random_crop(obs.clone())
		obs = augmentations.random_crop(obs)
		next_obs = augmentations.random_crop(next_obs)

		return obs, actions, rewards, next_obs, not_dones, pos

	def sample_drq(self, n=None, pad=4):
		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		obs = augmentations.random_shift(obs, pad)
		next_obs = augmentations.random_shift(next_obs, pad)

		return obs, actions, rewards, next_obs, not_dones
	
	def aux_sample(self, n=None, pad=4):
		idxs=self._get_aux_idxs(n)
		obs, _ = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		obs = augmentations.random_shift(obs, pad)
		return obs

	def sample(self, n=None):
		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		obs = augmentations.random_crop(obs)
		next_obs = augmentations.random_crop(next_obs)

		return obs, actions, rewards, next_obs, not_dones
	
	def sample_cpc(self,pad=4):

		start = time.time()
		idxs = np.random.randint(
			0, self.capacity if self.full else self.idx, size=self.batch_size
		)
	
		# obses = self.obses[idxs]
		# next_obses = self.next_obses[idxs]
		obses, next_obses = self._encode_obses(idxs)
		pos = obses.copy()
	
		obses = torch.as_tensor(obses, device=self.device).float()
		next_obses = torch.as_tensor(
			next_obses, device=self.device
		).float()
		actions = torch.as_tensor(self.actions[idxs], device=self.device)
		rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
		not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

		pos = torch.as_tensor(pos, device=self.device).float()

		obses = augmentations.random_shift(obses, pad)
		next_obses = augmentations.random_shift(next_obses, pad)
		pos = augmentations.random_shift(pos, pad)
		
		cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
						time_anchor=None, time_pos=None)

		return obses, actions, rewards, next_obses, not_dones, cpc_kwargs


class LazyFrames(object):
	def __init__(self, frames, extremely_lazy=True):
		self._frames = frames
		self._extremely_lazy = extremely_lazy
		self._out = None

	@property
	def frames(self):
		return self._frames

	def _force(self):
		if self._extremely_lazy:
			return np.concatenate(self._frames, axis=0)
		if self._out is None:
			self._out = np.concatenate(self._frames, axis=0)
			self._frames = None
		return self._out

	def __array__(self, dtype=None):
		out = self._force()
		if dtype is not None:
			out = out.astype(dtype)
		return out

	def __len__(self):
		if self._extremely_lazy:
			return len(self._frames)
		return len(self._force())

	def __getitem__(self, i):
		return self._force()[i]

	def count(self):
		if self.extremely_lazy:
			return len(self._frames)
		frames = self._force()
		return frames.shape[0]//3

	def frame(self, i):
		return self._force()[i*3:(i+1)*3]


def count_parameters(net, as_int=False):
	"""Returns total number of params in a network"""
	count = sum(p.numel() for p in net.parameters())
	if as_int:
		return count
	return f'{count:,}'

# 更改 begin
def make_heads(qkv, n_heads):
	shp = (qkv.size(0), qkv.size(1), n_heads, -1)
	return qkv.reshape(*shp).transpose(1, 2)


def multi_head_attention(q, k, v, mask=None):
	# q shape = (B, n_heads, 1/N, H/key_dim)
	# k,v shape = (B, n_heads, N, key_dim)
	# mask.shape = (B,N)

	B, n_heads, n, key_dim = q.shape

	# score.shape = (B, n_heads, n, N)
	score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(q.size(-1))

	if mask is not None:
		score += mask[:, None, None, :].expand_as(score)

	shp = [q.size(0), q.size(-2), q.size(1) * q.size(-1)]
	attn = torch.matmul(F.softmax(score, dim=3), v).transpose(1, 2)
	return attn.reshape(*shp)

def permute_data(patch_embeddings):
	"""
	:param patch_embeddings: output from ViTEncoder
	:return: shuffled patch_embeddings
	"""
	B, N, _ = patch_embeddings.shape
	# produce the random_index_mat and true label
	random_matrix = torch.rand((B, N))
	random_index = torch.argsort(random_matrix, dim=1)
	true_label=torch.argsort(random_index,dim=1)
	# shuffle
	for _ in range(len(patch_embeddings.shape[2:])):
		random_index = random_index[..., None]
	if torch.is_tensor(patch_embeddings):
		random_index = random_index.repeat(1, 1, *patch_embeddings.shape[2:]).to(patch_embeddings.device)
		patch_embeddings = patch_embeddings.gather(1, random_index)

	return patch_embeddings, true_label.cuda()
# 更改 end

def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones
    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs


class Intensity(torch.nn.Module):
	def __init__(self, scale):
		super().__init__()
		self.scale = scale

	def forward(self, x):
		r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
		noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
		return x * noise