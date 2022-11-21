import tensorflow as tf
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

# print(sess)

# with tf.compat.v1.Session() as sess:
#   devices = sess.list_devices()
#   print(devices)

# print("\n\n\n")
# print(tf.test.is_gpu_available())
# print(tf.test.gpu_device_name())

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)

# with tf.compat.v1.Session() as sess:
#     print (sess.run(c))

tf.config.list_physical_devices('GPU')