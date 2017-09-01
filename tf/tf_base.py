import tensorflow as tf
import numpy as np
import time
from easydict import EasyDict as edict
from os import path as osp
import matplotlib.pyplot as plt
import shutil
import functools

from pyhelper_fns import vis_utils
from pyhelper_fns import path_utils

class NetBase(object):
  def __init__(self, rand_seed=3, grad_clip=None, 
               net_prms=None):
    """
     net_prms:
      net_name
      batch_size
      weight_decay
    """
    self._ph = edict()
    #tf.reset_default_graph()
    self.rand_seed = rand_seed
    self.net_prms  = net_prms
    tf.set_random_seed(rand_seed)
    self._summary_writer = None
    self.grad_clip = grad_clip
    #Useful names
    self._names = {}
    self._names['summary'] = {}
    self._names['summary']['general']  = {} 
    self._names['summary']['general']['scalar']     = 'GENERAL_SUMMARIES_SCALAR'
    self._names['summary']['general']['histogram']  = 'GENERAL_SUMMARIES_HISTOGRAM'
    self._names['summary']['train']    = 'TRAIN_SUMMARIES'
    self._names['summary']['val']      = 'VAL_SUMMARIES'
    #Sumarries
    self._train_scalar_summary_ops = []
    self._train_summary_ops = []
    self._val_summary_ops = []
    #Account for params and grads that have already been added
    self._added_param_smmry = []
    self._added_grad_smmry  = []
    self._ops = edict()
    self._ops.init  = []
    self._ops.train = []
    self._ops.val   = []
    self._ops.loss  = []
    self._ops.acc   = [] 
    self._ops.other = {} 
    #List of weight decay variable
    self._wd_list   = []
    self.loss_finalized = False
 
  @property
  def weight_decay(self):
    return self.net_prms['weight_decay']
 
  @property
  def train_summary_ops(self):
    return self._train_summary_ops

  @property
  def train_scalar_summary_ops(self):
    return self._train_scalar_summary_ops

  @property
  def val_summary_ops(self):
    return self._val_summary_ops

  @property
  def train_ops(self):
    return self._ops.train

  @property
  def val_ops(self):
    return self._ops.val

  @property
  def init_ops(self):
    return self._ops.init

  @property
  def loss_ops(self):
    return self._ops.loss
  
  @property
  def acc_ops(self):
    return self._ops.acc

  @property
  def total_loss(self):
    assert self.loss_finalized, 'First finalize the loss'
    return self._total_loss 

  def other_ops_by_name(self, name):
    """
    fetch other ops by name
    """
    assert name in self._ops.other.keys()
    return self._ops.other[name]
 
  def to_list(self, op):
    if type(op) not in [list, tuple]:
      op = [op]
    return op
    
  def append_ops(self, ops, list_name):
    for op in self.to_list(ops):
      self._ops[list_name].append(op)
 
  def append_weight_decay(self, name):
    if name not in self._wd_list:
      self._wd_list.append(name) 
 
  def close(self):
    pass

  def __del__(self):
    self.close()

  def make_wd_loss_op(self):
    if self.weight_decay is None:
      self._ops.other['wd'] = []
      return []
    else:
      vrs = [v for v in tf.trainable_variables() if v.name in \
                    self._wd_list]
      with tf.variable_scope("weight_decay") as scope:
        w_loss = tf.add_n([tf.reduce_sum(tf.mul(v, v))\
                   for v in vrs])
        w_loss = 0.5 * self.weight_decay * w_loss
      self._ops.other['wd'] = [w_loss]
      return [w_loss] 

  def finalize_losses(self):
    if self.loss_finalized:
      raise Exception('LOSS HAS ALREADY BEEN FINALIZED')
    w_loss = self.make_wd_loss_op()
    self._ops['loss'] = self._ops['loss'] + w_loss
    self._total_loss = tf.add_n(self._ops['loss'], 'total_loss')
    self.loss_finalized = True

  def setup_learner(self):
    """
      Setup the optimizer for the agent
    """
    pass

  def merge_train_summary_ops(self):
    """Returns the list of summary ops during training """
    general_scalar_smmry = tf.summary.merge_all(key=self._names['summary']['general']['scalar'])
    general_hist_smmry = tf.summary.merge_all(key=self._names['summary']['general']['histogram'])
    train_smmry   = tf.summary.merge_all(key=self._names['summary']['train'])
    if general_scalar_smmry is not None:
      self._train_summary_ops.append(general_scalar_smmry)
      self._train_scalar_summary_ops.append(general_scalar_smmry)
    self._train_summary_ops.append(general_hist_smmry)
    if train_smmry is not None:
      self._train_summary_ops.append(train_smmry)
    #Scalar summary ops
    #self._train_scalar_summary_ops.append(train_smmry)

  def merge_val_summary_ops(self):
    val_smmry = tf.summary.merge_all(key=self._names['summary']['val'])
    if val_smmry is not None:
      self._val_summary_ops.append(val_smmry)

  def add_to_trainval_summary(self, ops, summary_type='scalar'):
    set_names = ['train', 'val']
    if not type(ops) in [list, tuple]:
      ops = [ops]
    for op in ops:
      for s in set_names:
        name = '%s_%s' % (s, op.name)
        if summary_type == 'scalar':
          print (name)
          tf.summary.scalar(name, op, collections=[self._names['summary'][s]])
        else:
          raise Exception('Only scalar summaries supported')

  def add_to_general_summary(self, ops, names=None, summary_type='scalar'):
    if not type(ops) in [list, tuple]:
      ops = [ops]
    if names is not None:
      names = [names]
      assert len(names) == len(ops)
    else:
      names = [None] * len(ops)
    for op, sm_name in zip(ops, names):
      if sm_name is None:
        sm_name = '%s_summary' % op.name
      if summary_type == 'scalar':
        tf.summary.scalar(op.name, op, 
              collections=[self._names['summary']['general']['scalar']])
      elif summary_type == 'histogram':
        print (op.name, op)
        tf.summary.histogram(op.name, op,
              collections=[self._names['summary']['general']['histogram']])
      else:
        raise Exception('Only scalar summaries supported')


  def get_train_feed_dict(self, ips):
    """Forms the feed_dict for normal training of the agent """
    pass

  def get_test_feed_dict(self, ips):
    """Forms the feed_dict for testing the agent """
    pass

  def add_param_summaries(self):
    for var in tf.trainable_variables():
      if var.op.name in self._added_param_smmry:
        print ('%s already added to summary' % var.op.name)
        continue
      tf.summary.histogram(var.op.name, var,
        collections=[self._names['summary']['general']['histogram']])
      self._added_param_smmry.append(var.op.name)

  def add_grad_summaries(self, grads=None):
    if grads is None:
      return
    for grad, var in grads:
      if var.op.name in self._added_grad_smmry:
        print ('%s already added to summary' % var.op.name)
        continue
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad,
          collections=[self._names['summary']['general']['histogram']])
        self._added_grad_smmry.append(var.op.name)

  def finalize(self):
    self.merge_train_summary_ops()
    self.merge_val_summary_ops()
    init_var  = tf.initialize_all_variables()
    tf.Graph.finalize(tf.get_default_graph())
    return init_var


class TrainManager(object):
  def __init__(self, nb, feed_dict_fn, log_freq=1000, snapshot_freq=3000,
      log_dir='/data0/pulkitag/tf_data/logs', 
      model_dir='/data0/pulkitag/tf_data/models',
      max_train_iter=100000, test_freq=500, 
      feed_dict_ph=[],
      train_feed_dict_args=[], val_feed_dict_args=[],
      print_freq=1, gpuFraction=0.3):
    """
      nb: NetDef
      feed_dict_fn: function that takes in (placeholders, obj)
                    where obj defines obj.sample_batch()
    """
    self.nb = nb
    self.log_freq  = log_freq
    self.snapshot_freq = snapshot_freq
    self.log_dir   = log_dir
    self.model_dir = model_dir
    self.max_train_iter = max_train_iter
    self.test_freq      = test_freq
    if model_dir is not None:
      path_utils.mkdir(self.model_dir)
      self._saver = tf.train.Saver(tf.global_variables(), 
                                   max_to_keep=5,
                                   keep_checkpoint_every_n_hours=2,
                                   name='save_model')
    else:
      self._saver = None
    #Feed dict relevant stuff
    self.feed_dict_fn = feed_dict_fn
    self.feed_dict_ph = feed_dict_ph
    self.train_feed_dict_args = train_feed_dict_args
    self.val_feed_dict_args   = val_feed_dict_args
    self.print_freq           = print_freq 
    #GPU options
    self.gpuFraction = gpuFraction
 
  def restore(self, sess):
    tf.Graph.finalize(tf.get_default_graph())
    ckpt = tf.train.get_checkpoint_state(osp.dirname(self.model_dir))
    fName = ckpt.model_checkpoint_path
    print ('Restoring from %s' % fName)
    self._saver.restore(sess, fName)
 
  def save_summaries(self, ops, i):
    for op in ops:
      self.sm_writer.add_summary(op, i)

  def print_losses(self, losses, setname='TRAIN'):
    for j, l in enumerate(losses):
      print ('%s, %s loss: %f' % (setname, self.nb.loss_ops[j].name, l))

  def print_accuracies(self, losses, setname='TRAIN'):
    for j, l in enumerate(losses):
      print ('%s, %s accuracy: %f' % (setname, self.nb.acc_ops[j].name, l))

  def delete_old_logs_and_models(self):
    if osp.exists(self.log_dir):
      shutil.rmtree(self.log_dir)     
      shutil.rmtree(self.model_dir)
    path_utils.mkdir(self.log_dir)
    path_utils.mkdir(self.model_dir)

  def setup_summary_writer(self):
    if self.log_dir is not None: 
      path_utils.mkdir(self.log_dir)
      self.sm_writer = tf.summary.FileWriter(self.log_dir, 
                    tf.get_default_graph())
    else:
      self.sm_writer = None

  def check_nan_inf(self, val):
    isNanInf = False
    isNanInf = isNanInf or np.isnan(val).any()
    if isNanInf:
      print ('NaN found')
    isNanInf = isNanInf or np.isinf(val).any()
    if isNanInf:
      print ('Inf found, only if NAN was not found before')
    return isNanInf
       
  def train(self):
    #Delete stuff from the last training run
    self.delete_old_logs_and_models()
    self.setup_summary_writer()
    
    #Merge train/val ops
    self.nb.merge_train_summary_ops()
    self.nb.merge_val_summary_ops()
    
    #Finalize the graph
    init_var  = tf.global_variables_initializer()
    tf.Graph.finalize(tf.get_default_graph())
    
    #Configure GPU options
    if self.gpuFraction is None:
      self.gpuFraction = 1.0
    gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpuFraction)
    
    with tf.Session(config=tf.ConfigProto(gpu_options=gpuOptions)) as sess:
      sess.run(init_var)
      init_ops = self.nb.init_ops
      if len(init_ops) > 0:
        sess.run(init_ops)

      t1 = time.time()
      for i in range(self.max_train_iter):
        log_smmry = False
        if np.mod(i, self.log_freq)==0:
          all_ops = self.nb.train_ops + self.nb.loss_ops + \
                    self.nb.acc_ops + self.nb.train_summary_ops
        else:
          all_ops = self.nb.train_ops + self.nb.loss_ops + self.nb.acc_ops
        #Number of train ops
        nt    = len(self.nb.train_ops)
        #Loss start and stop ops
        nl_st, nl_en = nt, nt + len(self.nb.loss_ops)
        #Accuracy start and stop ops
        na_st, na_en = nl_en, nl_en + len(self.nb.acc_ops)
        ns_st        = na_en
        delta_t   = time.time() - t1

        #Run the network
        feed_dict = self.feed_dict_fn(self.feed_dict_ph,
                             *self.train_feed_dict_args)
        res       = sess.run(all_ops, feed_dict=feed_dict)
        
        #Check if any of the losses went to NaN/Inf
        check = [self.check_nan_inf(v) for v in res[nl_st:nl_en]] 
        assert not np.array(check).any(), 'NaNs or Infs found'
        if np.mod(i, self.print_freq)==0:
          print ('Iter: %d, Total time: %f, time per iter %f' % \
            (i,delta_t, delta_t / max(i,1)))
          #Print losses
          self.print_losses(res[nl_st:nl_en], 'TRAIN')
          #Print accuracy
          self.print_accuracies(res[na_st:na_en], 'TRAIN')
        #Store summaries
        if np.mod(i, self.log_freq) == 0:
          self.save_summaries(res[ns_st:], i)
        #Store the model if required
        if np.mod(i, self.snapshot_freq) == 0:
          self._saver.save(sess, self.model_dir, global_step=i)
        
        #Test the model if required
        if np.mod(i, self.test_freq) == 0:
          print ('TESTING ...')
          all_ops = self.nb.loss_ops + \
                    self.nb.acc_ops + self.nb.val_summary_ops
          feed_dict = self.feed_dict_fn(self.feed_dict_ph, 
                        *self.val_feed_dict_args)
          res       = sess.run(all_ops, feed_dict=feed_dict)
          nl_st, nl_en = 0, len(self.nb.loss_ops)
          na_st, na_en = nl_en, nl_en + len(self.nb.acc_ops)
          ns_st        = na_en
          #Print losses
          self.print_losses(res[nl_st:nl_en], 'VAL')
          #Print accuracy
          self.print_accuracies(res[na_st:na_en], 'VAL')
          #Save summaries 
          self.save_summaries(res[ns_st:], i)
 

def get_weight_initializer(init_type='fanin', **kwargs):
  if init_type == 'fanin':
    ip_dims = kwargs['ip_dims']
    mn = 1.0 * (-1/np.sqrt(np.float(ip_dims)))
    mx = (-mn)
    print ("Weight Initializers:", mn, mx)
    w_init = tf.random_uniform_initializer(mn, mx)    
  else:
    raise Exception('init_type %s not recognized' % init_type)
  return w_init


def get_conv_layer(ip, sz, op_channels, stride=1, name='conv1', 
              activation='relu', init_type='fanin', padding='VALID',
              nb=None):
 with tf.variable_scope(name) as scope:
   ip_channels = int(ip.get_shape()[3])
   ip_dims     = sz * sz * ip_channels
   conv_init   = get_weight_initializer(init_type=init_type, **{'ip_dims': ip_dims})
   w = tf.get_variable("w", [sz, sz, ip_channels, op_channels], initializer=conv_init) 
   nb.append_weight_decay(w.name)
   b = tf.get_variable("b", [op_channels], initializer=tf.constant_initializer(0.))
   op = tf.nn.bias_add(tf.nn.conv2d(ip, w, [1, stride, stride, 1],
            padding=padding, use_cudnn_on_gpu=None), b) 
   if activation is not None:
    print ('Activation in CONV %s' % activation)
    op_fn = getattr(tf.nn, activation)
    op    = op_fn(op)
   if nb is not None:
     nb.add_to_general_summary(op, summary_type='histogram') 
   #op = tf.Print(op, [op], op.name)
 return op


class ConvLayerStack(object):
  def __init__(self, szs, op_channels, strides=1,
                name='conv', activation='relu', nb=None, **kwargs):
    """
      nb: object of type NetBase
    """
    assert len(szs) == len(op_channels)
    if not type(strides) in [list, tuple]:
      strides = len(szs) * [strides]
    self.szs         = szs
    self.op_channels = op_channels
    self.strides     = strides
    self.activation  = activation
    self.name        = name
    self.nb          = nb
    self.conv_args   = kwargs 
 
  def get(self, ip):
    op = ip
    count = 1
    for sz, opc, stride in zip(self.szs, self.op_channels, self.strides):
      op = get_conv_layer(op, sz, opc, stride,
             name='%s_%d' % (self.name, count), nb=self.nb,
             activation=self.activation, **self.conv_args)
      count += 1
    return op


def get_fc_layer(ip, op_channels, name='fc1', activation='relu',
                      init_type='fanin', nb=None):
  ip_dims = int(ip.get_shape()[1])
  with tf.variable_scope(name) as scope:
    w_init   = get_weight_initializer(init_type=init_type, **{'ip_dims': ip_dims})
    w = tf.get_variable("w", [ip_dims, op_channels], initializer=w_init) 
    nb.append_weight_decay(w.name)
    b = tf.get_variable("b", [op_channels], initializer=tf.constant_initializer(0.))
    #b = tf.get_variable("b", [op_channels], initializer=w_init)
    op = tf.nn.bias_add(tf.matmul(ip, w), b)  
    if activation is not None:
      op_fn = getattr(tf.nn, activation)
      op    = op_fn(op)
    if nb is not None:
      nb.add_to_general_summary(op, summary_type='histogram') 
    #op = tf.Print(op, [op], op.name)
  return op


class FCLayerStack(object):
  def __init__(self, op_channels, name='fc', activation='relu',
               init_type='fanin', nb=None, is_drop=False):
    assert type(op_channels) is list
    if not type(is_drop) is list:
      is_drop = [is_drop]
    self.op_channels = op_channels
    self.name        = name
    self.activation  = activation
    self.init_type   = 'fanin'
    self.nb          = nb
    self.is_drop     = is_drop
  
  def get(self, ip):
    op = ip
    count = 1
    for opc, drp in zip(self.op_channels, self.is_drop):
      op = get_fc_layer(op, opc, activation=self.activation, 
                        init_type=self.init_type,
                        name='%s_%s' % (self.name, count), nb=self.nb)
      if drp:
        op = tf.nn.dropout(op, 0.5, name='%s_%s_drop' %(self.name, count))  
      count += 1
    return op


def create_encoder(ims, convLayers, fcLayers, 
        concatMode='lateFusion'):
  """
  Args:
    ims       : list of images
    convLayers: object of type ConvLayerStack
    fcLayers  : object of type FCLayerStack
    concatMode: lateFusion -- concatenate the features after
                  processing images through convLayers
                earlyFusion -- concatenate the images and then
                  process through convLayers
  """
  assert type(ims) in [list, tuple]
  conv_stack = tf.make_template("convs", convLayers.get)
  fc_stack   = tf.make_template("fcs", fcLayers.get)

  if convLayers.nb.net_prms is not None:
    #Convolutional processing
    if convLayers.nb.net_prms['mean_subtract']:
      ims = [im - 128. for im in ims]
    if convLayers.nb.net_prms['im_nrmlz']:
      ims = [im / 128. for im in ims]
    
  #Concatenat the images/features
  if concatMode is 'lateFusion':
    ops = [conv_stack(im) for im in ims]
    ops = tf.concat(ops, 3)
  elif concatMode is 'earlyFusion':
    imStack = tf.concat(ims, 3)
    print ('imStack', imStack.get_shape())
    ops = conv_stack(ims)
  else:
    raise Exception('concatMode {0} is invalid'.format(concatMode))
  
  #Reshape the ops to be consistent with the fc-layers
  op_sz = functools.reduce(lambda x, y: int(x) * int(y), ops.get_shape()[1:])
  with tf.variable_scope('layer_reshape') as scope:
    ops = tf.reshape(ops, [-1, op_sz])
  #Fully connected processing 
  fc_ops = fc_stack(ops)
  return fc_ops