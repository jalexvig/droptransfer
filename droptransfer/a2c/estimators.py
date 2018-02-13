import tensorflow as tf


def build_mlp(
    input_placeholder,
    output_size,
    n_layers=2,
    size=64,
    activation=tf.tanh,
    output_activation=None
):

    l = input_placeholder

    for i in range(n_layers):
        l = tf.layers.dense(l, size, activation=activation, name='hidden_layer%i' % i)

    output_layer = tf.layers.dense(l, output_size, activation=output_activation, name='output_layer')

    return output_layer


class Estimator(object):

    def __init__(self, obs_dim):

        self.observations = tf.placeholder(shape=[None, obs_dim], name="observations", dtype=tf.float32)
        self.targets = tf.placeholder(shape=[None], name='targets', dtype=tf.float32)


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

            tf.summary.scalar('loss', self.loss)


class ValueEstimator(Estimator):

    def __init__(self, obs_dim):

        super(ValueEstimator, self).__init__(obs_dim)

        with tf.variable_scope('value_net'):

            self.predicted_values = tf.reshape(build_mlp(self.observations, 1), [-1])

            self.loss = tf.nn.l2_loss(self.predicted_values - self.targets)

            self.update_op = tf.train.AdamOptimizer().minimize(self.loss)

            tf.summary.scalar('loss', self.loss)
