import tensorflow as tf

path = "/tmp/q4_2_10"

def cnn_model_fn(features, labels, mode):

    input_layer = tf.reshape(features["x"], [-1, 96, 96, 3])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=128,
        kernel_size=[2, 2],
        padding="same",
        activation=tf.nn.relu,
        name="conv1")

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="pool1")

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[2, 2],
        padding="same",
        activation=tf.nn.relu,
        name="conv2")

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="pool2")

    pool2_flat = tf.reshape(pool2, [-1, 24 * 24 * 128])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name="dense")

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10, name='logits')

    predictions = {
        "classes": tf.argmax(logits, 1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")

    }

        names =['conv1','conv2','dense']

        for name in names:
          with tf.variable_scope(name, reuse=True):
            w = tf.get_variable('kernel')
            tf.summary.histogram(name, w)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)

        tf.summary.scalar('train_loss', loss)

        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        summary_hook = tf.train.SummarySaverHook(2, output_dir=path + '/log',
                                                 summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                          training_hooks=[summary_hook])
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    