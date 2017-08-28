# -*- coding: utf-8 -*-
import requests
import tensorflow as tf
import random


def digitize(data=list, listInput=list, listOutput=list, listDict=dict):
    outField = listOutput.pop()

    for x, i in enumerate(data):
        li = []
        lo = [0]
        for j in listDict.keys():
            if len(listDict[j]) == 0:
                listDict[j].append('0')
            try:
                if i['_source'][j] not in listDict[j]:
                    listDict[j].append(i['_source'][j])
                if j != outField:
                    li.append(listDict[j].index(i['_source'][j]))
                else:
                    lo[0] = listDict[j].index(i['_source'][j])
            except KeyError:
                li = None
                break
        if li:
            listInput.append(li)
            listOutput.append(lo)


def fetch():
    url = 'http://172.26.128.9:9200/logstash-jllog/_search?size=1000&filter_path=hits.hits._source'
    try:
        data = requests.post(url, auth=requests.auth.HTTPBasicAuth('elastic', 'changeme'))
    except IOError, e:
        print e
        exit()
    try:
        data = data.json()['hits']['hits']
        listInput = []
        listOutput = ['person_id']
        listDict = {"cmpip": [], "cmpname": [], "dept_name": [], "dnshz": [], "gzgw_name": [], "log_origin_ip": [],
                    "log_origin_name": [], "person_id": [], "person_name": [], "store": [], "sysname": []}
        digitize(data, listInput, listOutput, listDict)
        return listInput, listOutput, listDict
    except KeyError, e:
        print e
        print data.status_code


def trans(data=list, listInput=list, listOutput=list, listDict=dict):
    outField = listOutput.pop()
    for x, i in enumerate(data):
        li = []
        lo = [0]
        for j in listDict.keys():
            try:
                if j != outField:
                    li.append(listDict[j].index(i['_source'][j]))
                else:
                    lo[0] = listDict[j].index(i['_source'][j])
            except KeyError:
                li = None
                break
        if li:
            listInput.append(li)
            listOutput.append(lo)


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def dataBatch(listInput, listOutput, batchsize=8):
    listInBatch = []
    listOutBatch = []
    for i in range(batchsize):
        rd = random.randrange(0, len(listInput), 1)
        listInBatch.append(listInput[rd])
        listOutBatch.append(listOutput[rd])
    return listInBatch, listOutBatch


listInput, listOutput, listDict = fetch()
xs = tf.placeholder(tf.float32, [None, 10])
ys = tf.placeholder(tf.float32, [None, 1])
l1 = add_layer(xs, 10, 20, activation_function=tf.nn.relu)
l2 = add_layer(l1, 20, 20, activation_function=tf.nn.sigmoid)
l3 = add_layer(l2, 20, 10, activation_function=tf.nn.tanh)
# l4 = add_layer(l3, 10, 10, activation_function=tf.nn.tanh)
prediction = add_layer(l3, 10, 1, activation_function=tf.nn.relu)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000000):
    # listInBatch, listOutBatch = dataBatch(listInput, listOutput, 10)
    sess.run(train_step, feed_dict={xs: listInput, ys: listOutput})
    if (i + 1) % 10000 == 0:
        # prediction_value = sess.run(prediction, feed_dict={xs: listInput})
        # print "预测值：", str(prediction_value), "预期值：", str(listOutput)
        print sess.run(loss, feed_dict={xs: listInput, ys: listOutput})
