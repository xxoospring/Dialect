import tensorflow as tf

iter_epoch = 100000
train_epoch = 10000
batch_size = 128
nDataTrain = 36000
nDataDev = 1500
record_root = '../dataset/'
n_class = 6
log_dir = './logdir/'
ckpt_dir = './ckpt/'


def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'mfcc': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([n_class], tf.float32)
        }
    )
    spec = tf.decode_raw(features['mfcc'], tf.float64)
    label = features['label']
    return spec, label


def get_dataset(fname):
    dataset = tf.data.TFRecordDataset(fname)
    return dataset.map(parser)  # use padded_batch method if padding needed


# training dataset
dataset_train = get_dataset([record_root+'part-0_train.tfrecord', record_root+'part-1_train.tfrecord'])
dataset_train = dataset_train.repeat(iter_epoch).shuffle(1000).batch(batch_size)  # make sure repeat is ahead batch

dataset_dev = get_dataset([record_root+'part-0_dev.tfrecord'])
dataset_dev = dataset_dev.repeat(iter_epoch).shuffle(1000).batch(batch_size)  # make sure repeat is ahead batch

iter_train = dataset_train.make_one_shot_iterator()
iter_dev = dataset_dev.make_one_shot_iterator()

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types, dataset_train.output_shapes)
x, y_ = iterator.get_next()


def model_dense(inputs, n_classes=n_class):
    # build model
    inputs = tf.reshape(inputs, (-1, 3008))
    net = tf.layers.dense(inputs, 128, activation=tf.nn.relu)
    net = tf.layers.dense(net, 256, activation=tf.nn.relu)
    net = tf.layers.dense(net, 512, activation=tf.nn.relu)
    net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
    net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
    pred = tf.layers.dense(net, n_classes)
    return pred


pred = model_dense(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=pred))
train_op = tf.train.GradientDescentOptimizer(1.0e-4).minimize(loss)
# evaluate
cross = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y_, 1))
acc = tf.reduce_mean(tf.cast(cross, tf.float32))


def summary_op(data_type='train'):
    tf.summary.scalar(data_type+'-loss', loss)
    tf.summary.scalar(data_type+'-accuracy', acc)
    return tf.summary.merge_all()


summ_op_train = summary_op()
summ_op_dev = summary_op('val')

saver = tf.train.Saver(max_to_keep=5,)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    handle_train, handle_dev = sess.run([iter_train.string_handle(), iter_dev.string_handle()])
    summ_writer = tf.summary.FileWriter(log_dir, sess.graph)
    for _ in range(train_epoch):
        __, summ_train = sess.run([train_op, summ_op_train], feed_dict={handle: handle_train})
        if not _ % 100:
            summ_writer.add_summary(summ_train, global_step=_)
            loss_train = sess.run(loss, feed_dict={handle: handle_train})
            loss_dev, accuracy, summ_dev = sess.run([loss, acc, summ_op_dev], feed_dict={handle: handle_dev})
            summ_writer.add_summary(summ_dev, global_step=_)
            print('Train Loss: %-10.8f\t\tDev Loss: %-10.8f\t\tAcc: %-7.5f' % (loss_train, loss_dev, accuracy))
