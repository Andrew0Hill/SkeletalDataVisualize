import numpy as np
import pandas as pd
class DoubleExponentialFilter(object):
	def __init__(self, alpha, gamma):
		self.alpha = alpha
		self.gamma = gamma

	def filter(self, y, filt_jitter=True):
		# Y is an ndarray of shape
		# (num_frames, num_joints, 3)
			
		s_0 = y[0]
		b_0 = y[1] - y[0]

		prev_s = s_0
		prev_b = b_0

		for i in range(1, y.shape[0]):
			if filt_jitter:
				# Get the difference between the current measurement and the previous measurement.
				prev_diff = y[i] - y[i-1]
				# Get the vector magnitude of the differences
				diff_magn = np.linalg.norm(prev_diff,axis=1)
				filt_idcs = np.where(diff_magn <= 0.05)
				y[i,filt_idcs[0]] = np.multiply(y[i,filt_idcs[0]], (diff_magn[filt_idcs].reshape(-1,1)/0.03)) + \
									np.multiply(y[i-1,filt_idcs[0]],1-(diff_magn[filt_idcs].reshape(-1,1)/0.03))
				
		
			s_t = self.alpha * y[i] + (1 - self.alpha) * (prev_s + prev_b)
			prev_b = self.gamma * (s_t - prev_s) + (1 - self.gamma) * prev_b
			y[i] = s_t
			prev_s = s_t
