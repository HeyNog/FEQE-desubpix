#------------------
# Author luzhongshan
# Time2019/5/15 21:38
#------------------
import torch


def de_subpix(y, DF):
	(b, c, h, w) = y.shape
	assert (h%DF == 0 and w%DF == 0), 'Input Shape (%d,%d) mismatch with Downsampling-Factor %d'%(h,w,DF)

	h1 = int(h / DF)
	w1 = int(w / DF)
	d = []
	for i in range(DF**2):
		d.append(torch.zeros((b, c, h1, w1)))

	for i in range(0, h1):
		for j in range(0, w1):
			for k in range(0, DF):
				for l in range(0, DF):
					d_idx = k*DF + l
					d[d_idx][:,:,i,j] = y[:,:,i*DF+k,j*DF+l]
	out = torch.cat(d, dim=1)

	return out
