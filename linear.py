# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:08:15 2018
@author: wmy
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#生成等差数列
train_x = np.linspace(-1, 1, 100)
#*train_x.shape=100
train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.3
plt.plot(train_x, train_y, 'ro', label='original data')
plt.legend()
plt.show()

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

Z = tf.multiply(X, W) + b

plotdata = {"batchsize":[], "loss":[]}

def moving_average(a,w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

cost = tf.reduce_mean(tf.square(Y-Z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

training_epochs = 20
display_step = 2
with tf.Session() as sess:
    sess.run(init)
    plotdata={"batchsize":[],"loss":[]}
    for epoch in range(training_epochs):
        for (x, y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X:x, Y:y})
            
        if epoch % display_step == 0:
            loss = sess.run(cost,feed_dict={X:train_x,Y:train_y})
            print("epoch:", epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    print("finished!")
    print("cost=",sess.run(cost, feed_dict={X:train_x,Y:train_y}),"W=",sess.run(W),"b=",sess.run(b))
         
    plt.plot(train_x, train_y, 'ro', label='original data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Fittedline')
    plt.legend()
    plt.show()
    
    plotdata["avgloss"]=moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('minibatch number')
    plt.ylabel('loss')
    plt.title('minibatch run vs. training loss')
    
    plt.show()
    
