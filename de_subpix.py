#------------------
# Author luzhongshan
# Time2019/5/15 21:38
# https://github.com/958099161/FEQE-desubpix
#------------------
import torch
import torch.nn as nn


def de_subpix(y, DF):
	'''
	Desubpixel Shuffle in FEQE.
	Args:
		y: The input tensor in the desubpixel shuffle, shape (b, c, h, w)
		DF: The downsampling factor of the desubpixel shuffle.

	Output:
		out: The output tensor of desubpixel shuffle, shape (b, DF^2*c, h//DF, w//DF)
	'''
	(b, c, h, w) = y.shape
	assert (h%DF == 0 and w%DF == 0), 'Input Shape (%d,%d) mismatch with Downsampling-Factor %d'%(h,w,DF)

	h1 = int(h / DF)
	w1 = int(w / DF)
	d = []
	for i in range(DF):
		for j in range(DF):
			d.append(y[:, :, i::DF, j::DF])

	out = torch.cat(d, dim=1)

	return out


class DeSubPixelShuffle(nn.Module):
	def __init__(self, DF):
		super(DeSubPixelShuffle, self).__init__()
		self.df = DF

	def forward(self, x):
		return de_subpix(x, self.df)

if __name__ == '__main__':
	a = torch.rand((1,1,24,24))
	print(a.shape)
	b = de_subpix(a, 4)
	print(b.shape)
	c = nn.functional.pixel_shuffle(b, 4)
	print(c.shape)
	print(((a-c)**2).sum())
