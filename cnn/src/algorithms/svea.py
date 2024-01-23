import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC
import wandb
import math


class SVEA(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.svea_alpha = args.svea_alpha
		self.svea_beta = args.svea_beta
		self.use_aux = args.use_aux
		self.aux_update_freq = args.aux_update_freq
		self.aux_batch_size = args.aux_batch_size
		self.log_interval = args.log_interval
		if args.aug:
			self.aug=augmentations.random_conv

	
		if self.use_aux:
			self.aux_jigsaw = m.Decoder(encoder_for_aux=self.aux_encoder, decoder_head=self.decoder_head, embedding_dim=args.embed_dim,
										n_heads=args.num_heads,
										tanh_clipping=args.tanh_clipping).cuda()
			self.aux_jigsaw_optimizer = torch.optim.Adam([{"params":self.aux_jigsaw.decoder_head.parameters(),"lr":args.decoder_head_lr},
      {"params":self.aux_jigsaw.encoder_for_aux.parameters()},{"params":self.aux_jigsaw.Wq_graph.parameters()},
      {"params":self.aux_jigsaw.Wq_first.parameters()},{"params":self.aux_jigsaw.Wq_last.parameters()},{"params":self.aux_jigsaw.v1},
      {"params":self.aux_jigsaw.v2},{"params":self.aux_jigsaw.Wk.parameters()},{"params":self.aux_jigsaw.Wv.parameters()},
      {"params":self.aux_jigsaw.logit_Wk.parameters()},{"params":self.aux_jigsaw.multi_head_combine.parameters()}], lr=args.aux_jigsaw_lr,
															betas=(args.aux_jigsaw_beta, 0.999))
			print("params of encoder_for_aux:",self.aux_jigsaw.encoder_for_aux.parameters())
			print("params of decoder_head:",self.aux_jigsaw.decoder_head.parameters())


		self.train()
		self.critic_target.train()

	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
									target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		if self.svea_alpha == self.svea_beta:
			obs = utils.cat(obs, augmentations.random_conv(obs.clone()))
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)

			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = (self.svea_alpha + self.svea_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.svea_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			obs_aug = augmentations.random_conv(obs.clone())
			current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
			critic_loss += self.svea_beta * \
				(F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)
		wandb.log({"critic_loss": critic_loss}, step=step)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_drq()

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

	
		if self.use_aux and (step%self.aux_update_freq)==0:
			self.update_aux_jigsaw(obs[:self.aux_batch_size], L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()


	def update_aux_jigsaw(self, observation, L, step):
		if self.aug is not None:
			observation = self.aug(observation.clone())
		patch_embeddings = self.aux_jigsaw.encoder_for_aux(observation)
		patch_embeddings = self.aux_jigsaw.decoder_head(patch_embeddings)
		# B,_,D=embeddings.shape
		fixed_content_cls = patch_embeddings.mean(dim=1,keepdim=True)
		fixed_content_cls = fixed_content_cls.unsqueeze(1)
		patch_embeddings, true_label = utils.permute_data(patch_embeddings=patch_embeddings)
		self.aux_jigsaw.reset(fixed_content_cls=fixed_content_cls, patch_embeddings=patch_embeddings)
		last_patch = None
		aux_loss_function = aux_loss()
		jigsaw_loss = 0.0
		for i in range(len(patch_embeddings[0])):
			probs = self.aux_jigsaw(last_patch)
			last_patch = torch.argmax(probs, dim=1)
			jigsaw_loss += aux_loss_function(probs, true_label[:, i])
		jigsaw_loss = jigsaw_loss / len(patch_embeddings[0])
		
		if L is not None:
			L.log("train_jigsaw/loss", jigsaw_loss, step)
		wandb.log({"aux_task_loss": jigsaw_loss}, step=step)
		
		self.aux_jigsaw_optimizer.zero_grad()
		jigsaw_loss.backward()
		self.aux_jigsaw_optimizer.step()


class aux_loss(torch.nn.Module):
	def __init__(self):
		super(aux_loss, self).__init__()

	def forward(self, probs, true_label):
		loss = -torch.sum(torch.gather(input=probs, dim=1, index=true_label.unsqueeze(1)))
		return loss / probs.shape[0]