import tensorflow as tf

from droptransfer import CONFIG


def build_mlp(
    input_placeholder,
    output_size,
    activation=tf.tanh,
    output_activation=None
):

    l = input_placeholder

    for i, size in enumerate(CONFIG.num_units):
        with tf.variable_scope('hidden%i' % i):
            l = tf.layers.dense(l, size, activation=activation)
            l = tf.nn.dropout(l, CONFIG.dropout_keep)

    with tf.variable_scope('output'):
        output_layer = tf.layers.dense(l, output_size, activation=output_activation)

    return output_layer


class Estimator(object):

    def __init__(self, obs_dim):

        self.observations = tf.placeholder(shape=[None, obs_dim], name="observations", dtype=tf.float32)
        self.targets = tf.placeholder(shape=[None], name='targets', dtype=tf.float32)

    def get_reset_ops(self):

        # TODO(jalex): Experiment with resetting biases

        reset_ops = []

        for i in range(len(CONFIG.num_units)):
            with tf.variable_scope('hidden%i' % i, reuse=True):
                weights = tf.get_variable('dense/kernel')

            mask_reset = tf.cast(tf.random_uniform(weights.shape) < CONFIG.permanent_keep, weights.dtype)

            # TODO(jalex): Use initializer? Right now just zeroing them out
            reset_ops.append(tf.assign(weights, weights * mask_reset))

        return reset_ops


class PolicyEstimator(Estimator):

    def __init__(self, obs_dim, num_actions):

        super(PolicyEstimator, self).__init__(obs_dim)

        self.actions = tf.placeholder(shape=[None], name="actions", dtype=tf.int32)

        with tf.variable_scope('policy_net'):
            logits = build_mlp(self.observations, num_actions)

            self.sampled = tf.reshape(tf.multinomial(logits, 1), [-1])

            sampled_logprobs = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.actions)

            self.loss = -tf.reduce_mean(tf.multiply(sampled_logprobs, self.targets), name='loss')
            self.update_op = tf.train.AdamOptimizer().minimize(self.loss)

            self.reset_ops = self.get_reset_ops()

            tf.summary.scalar('loss', self.loss)


class ValueEstimator(Estimator):

    def __init__(self, obs_dim):

        super(ValueEstimator, self).__init__(obs_dim)

        with tf.variable_scope('value_net'):

            self.predicted_values = tf.reshape(build_mlp(self.observations, 1), [-1])

            self.loss = tf.nn.l2_loss(self.predicted_values - self.targets)

            self.update_op = tf.train.AdamOptimizer().minimize(self.loss)

            self.reset_ops = self.get_reset_ops()

            tf.summary.scalar('loss', self.loss)
