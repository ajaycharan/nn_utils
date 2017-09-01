import tensorflow as tf
import numpy as np
from nn_utils.tf import tf_base

class DataSampler(object):
	def sample_batch(self, ph, *args):
		#print (args)
		#print (type(ph), ph)
		out = {p: np.random.randn(32,64,64,3) for p in ph}
		return out

def test_simple_siamese():
	tf.reset_default_graph()
	imPh   = [tf.placeholder(tf.float32, [None, 64, 64, 3]) for i in range(2)]
	nb     = tf_base.NetBase()

	#Form the network architecture
	convLayers = tf_base.ConvLayerStack([5, 3, 3], [32, 32, 32],
								strides=[2, 1, 2], nb=nb)
	fcLayers   = tf_base.FCLayerStack([128, 128], nb=nb)
	op         = tf_base.create_encoder(imPh, convLayers, fcLayers)
	nb.append_ops(op, 'train')
	ds     = DataSampler()
	print (tf.global_variables())
	tm     = tf_base.TrainManager(nb, ds.sample_batch, 
						feed_dict_ph=imPh)
	tm.train()
