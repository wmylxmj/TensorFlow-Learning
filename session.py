# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 16:31:13 2018

@author: wmy
"""

import tensorflow as tf

#使用tf.constant定义一个字符串常量
hellow = tf.constant('Hellow, TensorFlow!')
#使用tf.Session建立一个会话
sess = tf.Session()
#运行会话
print(sess.run(hellow))
#关闭会话
sess.close()

#with session
a = tf.constant(3)
b = tf.constant(4)
#build a session named 'sess'
with tf.Session() as sess:
    print('相加： %i' % sess.run(a+b))
    print('相乘： %i' % sess.run(a*b))
    
#withsessionfeed
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a, b)
with tf.Session() as sess:
    print("add: %i" % sess.run(add, feed_dict={a:3,b:4}))
    print("mul: %i" % sess.run(mul, feed_dict={a:3,b:4}))
    print(sess.run([add, mul], feed_dict={a:3,b:4}))
    
