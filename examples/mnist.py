# coding=utf-8
# Copyright 2021 The Mesh TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MNIST using Mesh TensorFlow and TF Estimator.

This is an illustration, not a good model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import mnist_dataset as dataset  # local file import
import tensorflow.compat.v1 as tf

from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_config  # pylint: disable=g-direct-tensorflow-import
#from tensorflow.python.tpu import tpu_estimator  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu

from tensorflow_estimator.python.estimator.tpu import tpu_estimator
from tensorflow_estimator.python.estimator import estimator as estimator_lib

import six
import mock


tf.flags.DEFINE_string("data_dir", "/tmp/mnist_data",
                       "Path to directory containing the MNIST dataset")
tf.flags.DEFINE_integer("data_train_size", 60000,
                        "Number of examples in the MNIST training dataset.")
tf.flags.DEFINE_integer("data_eval_size", 10000,
                        "Number of examples in the MNIST validation dataset.")
tf.flags.DEFINE_string("model_dir", "/tmp/mnist_model", "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 200,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("hidden_size", 512, "Size of each hidden layer.")
tf.flags.DEFINE_integer("train_epochs", 40, "Total number of training epochs.")

tf.flags.DEFINE_integer('iterations', -1,
                        'Number of iterations per training loop.')

tf.flags.DEFINE_integer("epochs_between_evals", 1,
                        "# of epochs between evaluations.")
tf.flags.DEFINE_integer("eval_steps", 0,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_string("mesh_shape", "b0:2;b1:2;b2:2", "mesh shape")
tf.flags.DEFINE_string("layout", "batch:b0;row_blocks:b1;col_blocks:b2",
                       "layout rules")

# Cloud TPU Cluster Resolvers
tf.flags.DEFINE_string(
    'tpu',
    default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

tf.flags.DEFINE_string(
    'gcp_project',
    default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

tf.flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')


FLAGS = tf.flags.FLAGS


def use_tpu():
  return True if FLAGS.tpu else False


def contrib_summary_scalar(name, value, step):
  return tf.identity(value, name=name) # punt for now



class _CapturedObject(object):
  """A placeholder to capture an object.

  This is useful when we need to capture a Python object in the Tensorflow
  control flow body function and use it outside the control flow.
  """

  def __init__(self):
    self._object = None
    self._captured = False

  def capture(self, o):
    if self._captured:
      raise RuntimeError(
          'InternalError: Object can capture only once. Please file bug.')

    self._captured = True
    self._object = o

  def get(self):
    if not self._captured:
      raise RuntimeError(
          'InternalError: Object is not captured properly before `get`. '
          'Please file bug.')
    return self._object


class _CkptLoaderHook(tf.estimator.SessionRunHook):
  """Load checkpoint right after the session started."""

  def after_create_session(self, session, coord):
    # pylint: disable=protected-access
    saver_collection = tf.get_collection(tf.GraphKeys.SAVERS)
    if saver_collection:
      saver = saver_collection[0]
      check_point = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
      if check_point:
        saver.restore(session, check_point)


def mnist_model(image, labels, mesh):
  """The model.

  Args:
    image: tf.Tensor with shape [batch, 28*28]
    labels: a tf.Tensor with shape [batch] and dtype tf.int32
    mesh: a mtf.Mesh

  Returns:
    logits: a mtf.Tensor with shape [batch, 10]
    loss: a mtf.Tensor with shape []
  """
  batch_dim = mtf.Dimension("batch", FLAGS.batch_size)
  row_blocks_dim = mtf.Dimension("row_blocks", 4)
  col_blocks_dim = mtf.Dimension("col_blocks", 4)
  rows_dim = mtf.Dimension("rows_size", 7)
  cols_dim = mtf.Dimension("cols_size", 7)

  classes_dim = mtf.Dimension("classes", 10)
  one_channel_dim = mtf.Dimension("one_channel", 1)

  x = mtf.import_tf_tensor(
      mesh, image, #tf.reshape(image, [FLAGS.batch_size, 4, 7, 4, 7, 1]),
      mtf.Shape(
          [batch_dim, row_blocks_dim, rows_dim,
           col_blocks_dim, cols_dim, one_channel_dim]))
  x = mtf.transpose(x, [
      batch_dim, row_blocks_dim, col_blocks_dim,
      rows_dim, cols_dim, one_channel_dim])

  # add some convolutional layers to demonstrate that convolution works.
  filters1_dim = mtf.Dimension("filters1", 16)
  filters2_dim = mtf.Dimension("filters2", 16)
  f1 = mtf.relu(mtf.layers.conv2d_with_blocks(
      x, filters1_dim, filter_size=[9, 9], strides=[1, 1], padding="SAME",
      h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim, name="conv0"))
  f2 = mtf.relu(mtf.layers.conv2d_with_blocks(
      f1, filters2_dim, filter_size=[9, 9], strides=[1, 1], padding="SAME",
      h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim, name="conv1"))
  x = mtf.reduce_mean(f2, reduced_dim=filters2_dim)

  # add some fully-connected dense layers.
  hidden_dim1 = mtf.Dimension("hidden1", FLAGS.hidden_size)
  hidden_dim2 = mtf.Dimension("hidden2", FLAGS.hidden_size)

  h1 = mtf.layers.dense(
      x, hidden_dim1,
      reduced_dims=x.shape.dims[-4:],
      activation=mtf.relu, name="hidden1")
  h2 = mtf.layers.dense(
      h1, hidden_dim2,
      activation=mtf.relu, name="hidden2")
  logits = mtf.layers.dense(h2, classes_dim, name="logits")
  if labels is None:
    loss = None
  else:
    labels = mtf.import_tf_tensor(
        mesh, labels, #tf.reshape(labels, [FLAGS.batch_size]),
        mtf.Shape([batch_dim]))
    loss = mtf.layers.softmax_cross_entropy_with_logits(
        logits, mtf.one_hot(labels, classes_dim), classes_dim)
    loss = mtf.reduce_mean(loss)
  return logits, loss


def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  #with mtf.utils.outside_all_rewrites():
  if True:

    scalars = {}


    def _add_summary(lowering, train_or_eval, tf_loss, scalars, global_step):
      """Add all summaries."""
      for k in scalars.keys():
        if not isinstance(scalars[k], tf.Tensor):
          scalars[k] = tf.cast(
              lowering.export_to_tf_tensor(scalars[k]), tf.float32)

      def _host_loss_summary(global_step, tf_loss, **scalars):
        """Add summary.scalar in host side."""
        gs = tf.cast(global_step, tf.int64)
        sum_loss = contrib_summary_scalar(
            '{}_loss'.format(train_or_eval), tf_loss, step=gs)
        sum_ops = [sum_loss.op]
        for description, tf_metric in six.iteritems(scalars):
          sum_metric = contrib_summary_scalar(
              '{}_{}'.format(train_or_eval, description), tf_metric, step=gs)
          sum_ops.append(sum_metric)
        with tf.control_dependencies(sum_ops):
          return tf.identity(tf_loss)

      if use_tpu():
        # Cast the global step to tf.int32, since
        # outside_compilation does not support tf.int64.
        tf_loss = tpu.outside_compilation(
            _host_loss_summary,
            tf.cast(global_step, tf.int32),
            tf_loss,
            **scalars)
      else:
        tf_loss = _host_loss_summary(
            tf.cast(global_step, tf.int32),
            tf_loss,
            **scalars)

      return tf_loss


    tf.logging.info("features = %s labels = %s mode = %s params=%s" %
                    (features, labels, mode, params))
    global_step = tf.train.get_global_step()
    graph = mtf.Graph()
    mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
    layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)
    # mesh_size = mesh_shape.size
    # mesh_devices = [""] * mesh_size
    # mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
    #     mesh_shape, layout_rules, mesh_devices)

    if use_tpu():
      ctx = params['context']
      num_hosts = ctx.num_hosts
      host_placement_fn = ctx.tpu_host_placement_function
      device_list = [host_placement_fn(host_id=t) for t in range(num_hosts)]
      tf.logging.info('device_list = %s' % device_list,)
      # TODO(ylc): Better estimation of replica cache size?
      replica_cache_size = 300 * 1000000  # 300M per replica
      # Worker 0 caches all the TPU binaries.
      worker0_mem = replica_cache_size * ctx.num_replicas
      devices_memeory_usage = [worker0_mem] + [0] * (num_hosts - 1)
      var_placer = mtf.utils.BalancedVariablePlacer(device_list,
                                                    devices_memeory_usage)
      mesh_devices = [''] * mesh_shape.size
      mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
          mesh_shape, layout_rules, mesh_devices, ctx.device_assignment)
    else:
      var_placer = None
      mesh_devices = [''] * mesh_shape.size
      mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
          mesh_shape, layout_rules, mesh_devices)
    #mesh = mtf.Mesh(graph, "my_mesh")
    mesh = mtf.Mesh(graph, 'my_mesh', var_placer)
    with mtf.utils.outside_all_rewrites():
      # Do not tpu_rewrite this part. Inside this unet, If you use Tensorflow,
      # instead of Mesh-Tensorflor, it will cause host to tpu send/rec.
      logits, loss = mnist_model(features, labels, mesh)
    

    if mode == tf.estimator.ModeKeys.TRAIN:
      var_grads = mtf.gradients(
          [loss], [v.outputs[0] for v in graph.trainable_variables])
      optimizer = mtf.optimize.AdafactorOptimizer()
      update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

    lowering = mtf.Lowering(graph, {mesh: mesh_impl})

    train_or_eval='train' if mode == tf.estimator.ModeKeys.TRAIN else 'eval'

    if mode != tf.estimator.ModeKeys.PREDICT:
      tf_loss = lowering.export_to_tf_tensor(loss)
      #tf.summary.scalar("loss", tf_loss)
      # tf_loss = _add_summary(
      #     lowering, train_or_eval, tf_loss, scalars, global_step)
    else:
      # for now, we can only export fully-replicated tensors.
      fully_replicated_logits = mtf.anonymize(logits)
      
      tf_logits = lowering.export_to_tf_tensor(fully_replicated_logits)
      

    if mode == tf.estimator.ModeKeys.TRAIN:
      tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
      tf_update_ops.append(tf.assign_add(global_step, 1))
      train_op = tf.group(tf_update_ops)

    with mtf.utils.outside_all_rewrites():
      master_to_slice_hook = mtf.MtfRestoreHook(lowering)
      if mode == tf.estimator.ModeKeys.TRAIN:

        saver = tf.train.Saver(
            tf.global_variables(),
            keep_checkpoint_every_n_hours=2,
            save_relative_paths=True)
            # sharded=True,
            # max_to_keep=10,
            # keep_checkpoint_every_n_hours=2,
            # defer_build=False, save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        saver_listener = mtf.MtfCheckpointSaverListener(lowering)
        slice_to_master_hook = tf.train.CheckpointSaverHook(
            FLAGS.model_dir,
            save_steps=1000,
            saver=saver,
            listeners=[saver_listener])

        if False:
          accuracy = tf.metrics.accuracy(
              labels=labels, predictions=tf.argmax(tf_logits, axis=1))

          # Name tensors to be logged with LoggingTensorHook.
          tf.identity(tf_loss, "cross_entropy")
          tf.identity(accuracy[1], name="train_accuracy")

          # Save accuracy scalar to Tensorboard output.
          #tf.summary.scalar("train_accuracy", accuracy[1])
          scalars['train_accuracy'] = accuracy[1]
          tf_loss = _add_summary(
              lowering, train_or_eval, tf_loss, scalars, global_step)

        # master_to_slice_hook must come before slice_to_master_hook
        if use_tpu():
          return tpu_estimator.TPUEstimatorSpec(
              tf.estimator.ModeKeys.TRAIN, loss=tf_loss, train_op=train_op,
              training_hooks=[master_to_slice_hook, slice_to_master_hook, tf.train.StepCounterHook(every_n_steps=10)])
        else:
          return tf.estimator.EstimatorSpec(
              tf.estimator.ModeKeys.TRAIN, loss=tf_loss, train_op=train_op,
              training_chief_hooks=[master_to_slice_hook, slice_to_master_hook, tf.train.StepCounterHook(every_n_steps=10)])

      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classes": tf.argmax(tf_logits, axis=1),
            "probabilities": tf.nn.softmax(tf_logits),
        }
        assert not use_tpu()
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            prediction_hooks=[master_to_slice_hook],
            export_outputs={
                "classify": tf.estimator.export.PredictOutput(predictions)
            })
      if mode == tf.estimator.ModeKeys.EVAL:
        tf_loss = _add_summary(
            lowering, train_or_eval, tf_loss, scalars, global_step)
        assert not use_tpu()
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=tf_loss,
            evaluation_hooks=[master_to_slice_hook],
            eval_metric_ops={
                "accuracy":
                tf.metrics.accuracy(
                    labels=labels, predictions=tf.argmax(tf_logits, axis=1)),
            })


def run_mnist():
  """Run MNIST training and eval loop."""
  session_config = None
  if use_tpu():
    # meta-optimizer drastically slows down startup time and has little benefit
    # when running on TPU.
    session_config = tf.ConfigProto(
        graph_options=tf.GraphOptions(
            rewrite_options=rewriter_config_pb2.RewriterConfig(
                disable_meta_optimizer=True)))

  if use_tpu():
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    iterations_per_loop = FLAGS.iterations
    if iterations_per_loop <= 0:
      iterations_per_loop = FLAGS.data_train_size // FLAGS.batch_size
    tf.logging.info("iterations_per_loop=%s" % iterations_per_loop)
    mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
    
    config = tpu_config.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=None,  # Disable the default saver
        save_checkpoints_secs=None,  # Disable the default saver
        log_step_count_steps=iterations_per_loop,
        save_summary_steps=iterations_per_loop,
        session_config=session_config,
        tpu_config=tpu_config.TPUConfig(
            num_shards=mesh_shape.size,
            iterations_per_loop=iterations_per_loop,
            #num_cores_per_replica=8 // mesh_shape.size,
            num_cores_per_replica=1,
            per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST))
    mnist_classifier = tpu_estimator.TPUEstimator(
        use_tpu=True,
        model_fn=model_fn,
        config=config,
        train_batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.batch_size)
      
    
    # mnist_classifier = tpu_estimator.TPUEstimator(
    #     model_fn=model_fn,
    #     model_dir=FLAGS.model_dir)

      # train_batch_size=batch_size,
      # eval_batch_size=batch_size,
      # predict_batch_size=batch_size,
      # use_tpu=use_tpu,
      # export_to_tpu=False,
    
  else:
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir)

  current_step = estimator_lib._load_global_step_from_checkpoint_dir(FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long
  current_epoch = current_step // iterations_per_loop
  tf.logging.info('Current step %d epoch %d', current_step, current_epoch)


  # Set up training and evaluation input functions.
  def train_input_fn(params=None):
    """Prepare data for training."""

    batch_size = params['batch_size'] if params is not None else FLAGS.batch_size

    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes use less memory. MNIST is a small
    # enough dataset that we can easily shuffle the full epoch.
    ds = dataset.train(FLAGS.data_dir)
    ds_batched = ds.cache().shuffle(buffer_size=50000).batch(batch_size, drop_remainder=True)
    ds_batched = ds_batched.map(lambda features, labels, batch_size=batch_size: (features.set_shape([batch_size, features.shape[-1]]) or labels.set_shape([batch_size]) or (tf.reshape(features, [batch_size, 4, 7, 4, 7, 1]), labels)))
    #tf.reshape(image, [FLAGS.batch_size, 4, 7, 4, 7, 1])

    # Iterate through the dataset a set number (`epochs_between_evals`) of times
    # during each training session.
    ds = ds_batched.repeat(FLAGS.epochs_between_evals)
    #ds = ds_batched.repeat()
    return ds

  def eval_input_fn(params=None):
    batch_size = params['batch_size'] if params is not None else FLAGS.batch_size
    return dataset.test(FLAGS.data_dir).batch(
        batch_size, drop_remainder=True).make_one_shot_iterator().get_next()

  import tqdm
  if use_tpu():
    # Train and evaluate model.
    # import tqdm
    # for epoch in tqdm.trange(FLAGS.train_epochs):
    #if True:
    for _ in tqdm.trange(current_epoch, FLAGS.train_epochs // FLAGS.epochs_between_evals):
      # eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
      # print("\nEvaluation results:\n\t%s\n" % eval_results)
      #mnist_classifier.train(input_fn=train_input_fn, max_steps=FLAGS.iterations)
      mnist_classifier.train(input_fn=train_input_fn, steps=iterations_per_loop)
  else:
    # Train and evaluate model.
    for _ in tqdm.trange(current_epoch, FLAGS.train_epochs // FLAGS.epochs_between_evals):
      mnist_classifier.train(input_fn=train_input_fn, hooks=None)
      eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
      print("\nEvaluation results:\n\t%s\n" % eval_results)



E = tpu_estimator

def _train_on_tpu_system(ctx, model_fn_wrapper, dequeue_fn):
  """Executes `model_fn_wrapper` multiple times on all TPU shards."""
  iterations_per_loop_var = E._create_or_get_iterations_per_loop()

  (single_tpu_train_step, host_call, captured_scaffold_fn,
   captured_training_hooks) = (
       model_fn_wrapper.convert_to_single_tpu_train_step(dequeue_fn))

  @E.tpu_function.on_device_training_loop
  def multi_tpu_train_steps_on_single_shard(replica_id):
    # `tpu.split_compile_and_shard()` splits and passes input for each
    # replica as an array. As so, correctly reshape the input to be a
    # scalar.
    replica_id = tf.reshape(replica_id, [])
    with E.tpu_context._TPUEstimatorReplicaContext(replica_id):  # pylint: disable=protected-access
      # outputs = training_loop.while_loop(
      #     lambda i, loss: i < iterations_per_loop_var,
      #     lambda i, loss: [i + 1, single_tpu_train_step(i)],
      #     inputs=[0, _INITIAL_LOSS])
      # return outputs[1:]
      return single_tpu_train_step(0)

  # Add input that represents id for each replica in sync so that
  # _TPUEstimatorReplicaContext can be correctly entered during
  # replicated computation.
  replica_id_inputs = []
  replica_id_inputs.append([tf.constant(i) for i in range(ctx.num_replicas)])

  (compile_op, loss) = E.tpu.split_compile_and_shard(
      multi_tpu_train_steps_on_single_shard,
      inputs=replica_id_inputs,
      num_shards=ctx.num_replicas,
      outputs_from_all_shards=False,
      device_assignment=ctx.device_assignment)

  loss = loss[0]
  return (compile_op, loss, host_call, captured_scaffold_fn,
          captured_training_hooks.get())



def main(_):
  #with mock.patch.object(tf, "glorot_uniform_initializer", mock_initializer):
  #with mock.patch.object(tpu_estimator, "_train_on_tpu_system", _train_on_tpu_system):
  if True:
    run_mnist()


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.logging.set_verbosity(tf.logging.DEBUG)
  tf.get_logger().setLevel('DEBUG')
  tf.app.run()
