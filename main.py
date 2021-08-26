import numpy as np
import sys,os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()
from tensorflow.compat.v1.keras.utils import to_categorical
from model import *
from buffer import ReplayBuffer
from config import *
from utils import *
from smac.env import StarCraft2Env
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.Session(config=config_tf)

alpha = float(sys.argv[1])

env = StarCraft2Env(map_name='3s_vs_5z')
env_info = env.get_env_info()
n_ant = env_info["n_agents"]
n_actions = env_info["n_actions"]
feature_space = env_info["obs_shape"]
state_space = env_info["state_shape"]
observation_space = feature_space + n_ant

buff = ReplayBuffer(capacity,state_space,observation_space,n_actions,n_ant)
agents = Agent(sess,observation_space,n_actions,state_space,n_ant,alpha)
eoi_net = intrisic_eoi(feature_space,n_ant)
get_eoi_reward = build_batch_eoi(feature_space,eoi_net,n_ant)

agents.update()

q = np.ones((n_ant,batch_size,n_actions))
next_a = np.ones((n_ant,batch_size,n_actions))
feature = np.zeros((batch_size,feature_space))
feature_positive = np.zeros((batch_size,feature_space))

f = open(sys.argv[1]+'_'+sys.argv[2]+'.txt','w')

def test_agent():

	episode_reward = 0
	win = 0

	for e_i in range(20):
		env.reset()
		obs = get_obs(env.get_obs(),n_ant)
		mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
		terminated = False

		while terminated == False:
			action = []
			acts = []

			outs = agents.acting([np.array([obs[i]]) for i in range(n_ant)])
			action=[]
			for i in range(n_ant):
				a = np.argmax(outs[i][0] - 9e15*(1 - mask[i]))
				acts.append(a)
		
			reward, terminated, winner = env.step(acts)
			if winner.get('battle_won') == True:
				win += 1
			episode_reward += reward
			obs = get_obs(env.get_obs(),n_ant)
			mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
	return episode_reward/20, win/20
		

while i_episode<n_episode:
	i_episode += 1
	if i_episode > 40:
		epsilon -= 0.005
		if epsilon < 0.05:
			epsilon = 0.05
	env.reset()
	obs = get_obs(env.get_obs(),n_ant)
	state = env.get_state()
	mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
	terminated = False
	episode_reward = 0
	win = 0

	while terminated == False:
		action = []
		acts = []

		outs = agents.acting([np.array([obs[i]]) for i in range(n_ant)])
		action=[]
		for i in range(n_ant):
			if np.random.rand() < epsilon:
				avail_actions_ind = np.nonzero(mask[i])[0]
				a = np.random.choice(avail_actions_ind)
			else:
				a = np.argmax(outs[i][0] - 9e15*(1 - mask[i]))
			acts.append(a)
			action.append(to_categorical(a,n_actions))

		reward, terminated, winner = env.step(acts)
		if winner.get('battle_won') == True:
			win = 1
		episode_reward += reward
		next_obs = get_obs(env.get_obs(),n_ant)
		next_state = env.get_state()
		next_mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
		buff.add(np.array(obs),action,reward,np.array(next_obs),state,next_state,mask,next_mask,terminated)
		obs = next_obs
		state = next_state
		mask = next_mask
	sum_reward += episode_reward
	sum_win += win

	if i_episode%200==0:
		log_r, log_w = test_agent()
		h = str(int(i_episode/200))+': '+sys.argv[1]+': '+sys.argv[2]+': '+str(sum_reward/200)+': '+str(sum_win/200)+': '+str(log_r)+': '+str(log_w)
		print(h)
		f.write(h+'\n')
		f.flush()
		sum_reward = 0
		sum_win = 0

	if i_episode<100:
		continue

	samples, positive_samples = buff.getObs(batch_size)
	feature_label = np.random.randint(0,n_ant,batch_size)
	for i in range(batch_size):
		feature[i] = samples[feature_label[i]][i][0:feature_space]
		feature_positive[i] = positive_samples[feature_label[i]][i][0:feature_space]
	sample_labels = to_categorical(feature_label,n_ant)
	positive_labels = eoi_net.predict(feature_positive,batch_size=batch_size)
	eoi_net.fit(feature,sample_labels+beta_1*positive_labels,batch_size=batch_size,epochs=1,verbose=0)

	for e in range(epoch):

		o, a, r, next_o, s, next_s, mask, next_mask, d = buff.getBatch(batch_size)

		q_q = agents.batch_q([o[i] for i in range(n_ant)])
		next_q_q = agents.batch_q_tar([next_o[i] for i in range(n_ant)])
		eoi_r = get_eoi_reward.predict([o[i][:,0:feature_space] for i in range(n_ant)],batch_size = batch_size)
		for i in range(n_ant):
			best_a = np.argmax(next_q_q[i] - 9e15*(1 - next_mask[i]), axis = 1)
			next_a[i] = to_categorical(best_a,n_actions)
			q[i] = q_q[i + n_ant]
			for j in range(batch_size):
				q[i][j][np.argmax(a[i][j])] = gamma*(1-d[j])*next_q_q[i + n_ant][j][best_a[j]] + eoi_r[i][j]
		agents.train_critics(o,q)

		q_target = agents.Q_tot_tar.predict([next_o[i] for i in range(n_ant)]+[next_a[i] for i in range(n_ant)]+[next_s],batch_size = batch_size)
		q_target = r + q_target*gamma*(1-d)
		agents.train_qmix(o, a, s, mask, q_target)

	if i_episode%5 == 0:
		agents.update()
