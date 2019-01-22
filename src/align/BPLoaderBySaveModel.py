import os
import time
from tensorflow.python.client import device_lib
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# 设置tf记录那些信息，这里有以下参数：
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
# 在Linux下，运行python程序前，使用的语句是$ export TF_CPP_MIN_LOG_LEVEL=2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

Model_path = "/Users/wanghx/Documents/01_DEV/10_2019/TFTest/saveBPModelSample/BPModel/"

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

print(" tensorflow v " + tf.__version__)



Model_path = "/Users/eric/Documents/01_DEV/02_ArchTest/Facenet/models/MTCNN/saved_model_dir_signature"
NUM_STEPS = 400
MINIBATCH_SIZE = 100

signature_key='mtcnn_signature'
pnet_input_key='pnet_inputs'
rnet_input_key='rnet_inputs'
onet_input_key='onet_inputs'
pnet_conv4_2_BiasAdd_output_key='pnet_conv4_2_BiasAdd_outputs'
pnet_prob1_output_key='pnet_prob1_outputs'
rnet_conv5_2_conv5_2_output_key='rnet_conv5_2_conv5_2_outputs'
rnet_prob1_output_key='rnet_prob1_outputs'

onet_conv6_2_conv6_2_output_key='onet_conv6_2_conv6_2_outputs'
onet_conv6_3_conv6_3_output_key='onet_conv6_3_conv6_3_outputs'
onet_prob1_output_key='onet_prob1_outputs'


def create_mtcnn(sess, saved_model_dir):
    # 载入和使用带有signature的模型
    meta_graph_def = tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
    # 从meta_graph_def中提取signatureDef对象
    signature = meta_graph_def.signature_def
    # 从signature中找出具体输入和输出的对象
    pnet_input_tensor_name = signature[signature_key].inputs[pnet_input_key].name
    rnet_input_tensor_name = signature[signature_key].inputs[rnet_input_key].name
    onet_input_tensor_name = signature[signature_key].inputs[onet_input_key].name
    pnet_conv4_2_BiasAdd_output_tensor_name = signature[signature_key].outputs[pnet_conv4_2_BiasAdd_output_key].name
    pnet_prob1_output_tensor_name = signature[signature_key].outputs[pnet_prob1_output_key].name
    rnet_conv5_2_conv5_2_output_tensor_name = signature[signature_key].outputs[rnet_conv5_2_conv5_2_output_key].name
    rnet_prob1_output_tensor_name = signature[signature_key].outputs[rnet_prob1_output_key].name
    onet_conv6_2_conv6_2_output_tensor_name = signature[signature_key].outputs[onet_conv6_2_conv6_2_output_key].name
    onet_conv6_3_conv6_3_output_tensor_name = signature[signature_key].outputs[onet_conv6_3_conv6_3_output_key].name
    onet_prob1_output_tensor_name = signature[signature_key].outputs[onet_prob1_output_key].name

    # 获取tensor并inference
    pnet_input = sess.graph.get_tensor_by_name(pnet_input_tensor_name)
    rnet_input = sess.graph.get_tensor_by_name(rnet_input_tensor_name)
    onet_input = sess.graph.get_tensor_by_name(onet_input_tensor_name)
    pnet_conv4_2_BiasAdd_output = sess.graph.get_tensor_by_name(pnet_conv4_2_BiasAdd_output_tensor_name)
    pnet_prob1_output = sess.graph.get_tensor_by_name(pnet_prob1_output_tensor_name)
    rnet_conv5_2_conv5_2_output = sess.graph.get_tensor_by_name(rnet_conv5_2_conv5_2_output_tensor_name)
    rnet_prob1_output = sess.graph.get_tensor_by_name(rnet_prob1_output_tensor_name)
    onet_conv6_2_conv6_2_output = sess.graph.get_tensor_by_name(onet_conv6_2_conv6_2_output_tensor_name)
    onet_conv6_3_conv6_3_output = sess.graph.get_tensor_by_name(onet_conv6_3_conv6_3_output_tensor_name)
    onet_prob1_output = sess.graph.get_tensor_by_name(onet_prob1_output_tensor_name)


    # lambda:匿名函数
    pnet_fun = lambda img: sess.run((pnet_conv4_2_BiasAdd_output, pnet_prob1_output), feed_dict={pnet_input: img})
    rnet_fun = lambda img: sess.run((rnet_conv5_2_conv5_2_output, rnet_prob1_output), feed_dict={rnet_input: img})
    onet_fun = lambda img: sess.run((onet_conv6_2_conv6_2_output, onet_conv6_3_conv6_3_output, onet_prob1_output),
                                    feed_dict={onet_input: img})
    return pnet_fun, rnet_fun, onet_fun



with tf.Session(graph=tf.Graph()) as sess:
    # tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], Model_path)
    # sess.run(tf.global_variables_initializer())
    # meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], Model_path)
    # print(meta_graph_def)
    create_mtcnn(sess,Model_path)