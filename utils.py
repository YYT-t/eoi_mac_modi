import numpy as np
def get_obs(obs,n_agent):
	for i in range(n_agent):
		index = np.zeros(n_agent)
		index[i] = 1
		obs[i] = np.hstack((obs[i],index))
	return obs
