import tensorflow as tf
import align.detect_face as detect_face
import os
from tensorflow.python.framework.graph_util import convert_variables_to_constants

# tf.InteractiveSession()让自己成为默认的session，用户不需指明哪个session运行情况下就可运行起来
# tf.InteractiveSession()来构建会话时，可以先构建一个session然后再定义操作
# 使用tf.Session()来构建会话，需要在会话构建之前定义好全部的操作（operation）然后再构建会话
sess = tf.InteractiveSession()

with tf.variable_scope("pnet"):
    data = tf.placeholder(tf.float32, (None, None, None, 3), "input")
    pnet = detect_face.PNet({"data": data})
    pnet.load("det1.npy", sess)

with tf.variable_scope("rnet"):
    data = tf.placeholder(tf.float32, (None, 24, 24, 3), "input")
    rnet = detect_face.RNet({"data": data})
    rnet.load("det2.npy", sess)

with tf.variable_scope("onet"):
    data = tf.placeholder(tf.float32, (None, 48, 48, 3), "input")
    onet = detect_face.ONet({"data": data})
    onet.load("det3.npy", sess)

# 将存储到 .npy文件中的网络模型参数转换成用 .bp文件存储的模型格式
"""
tf模型导出为单个文件（同时包含模型架构定义与权重）
利用tf.train_write_graph()默认情况下只导出了网络的定义（无权重）
利用tf.train_Saver().Save()导出文件graph_def与权重分离的，graph_def没有包含网络中的Variable值
（通常情况只存储了权重），但却包含了constant值，如果把Variable转换成constant，
可达到使用一个文件同时存储网络架构与权重的目标
convert_variables_to_constants函数会将计算图中的变量取值以常量的形式保存，
在保存模型文件的时候只是导出了GraphDef部分，GraphDef保存了从输入到输出的计算过程
保存的时候通过convert_variables_to_constants函数来指定保存的节点名称而不是张量的名称
比如：“add:0”是张量的名称，而"add"表示的是节点的名称。
"""
#
constant_graph = convert_variables_to_constants(sess, sess.graph_def,
                                                ["pnet/input", "rnet/input", "onet/input",
                                                 "pnet/conv4-2/BiasAdd", "pnet/prob1",
                                                 "rnet/conv5-2/conv5-2", "rnet/prob1",
                                                 "onet/conv6-2/conv6-2", "onet/conv6-3/conv6-3",
                                                 "onet/prob1"])
with tf.gfile.FastGFile("face_detect.pb", mode="wb") as f:
    f.write(constant_graph.SerializeToString())

"""
saved_model模块主要用于TensorFlow serving，TF serving是一个将训练好的模型部署
至生产环境的系统，主要优点在于：保持server端与API不变情况下部署新的算法或进行
实验，同时还有很高的性能。
"""
# 构造SavedModelBuilder对象，初始化方法只需要传入传入用于保存模型的目录名，目录不用预先创建
builder = tf.saved_model.builder.SavedModelBuilder("/Users/eric/Documents/01_DEV/02_ArchTest/Facenet/models/MTCNN/saved_model_dir_signature/")

# 输入（三个）与输出（七个）tensor获取
pnet_input_tensor = tf.get_default_graph().get_tensor_by_name("pnet/input:0")
rnet_input_tensor = tf.get_default_graph().get_tensor_by_name("rnet/input:0")
onet_input_tensor = tf.get_default_graph().get_tensor_by_name("onet/input:0")

pnet_conv4_2_BiasAdd_output_tensor = tf.get_default_graph().get_tensor_by_name("pnet/conv4-2/BiasAdd:0")
pnet_prob1_output_tensor = tf.get_default_graph().get_tensor_by_name("pnet/prob1:0")
rnet_conv5_2_conv5_2_output_tensor = tf.get_default_graph().get_tensor_by_name("rnet/conv5-2/conv5-2:0")
rnet_prob1_output_tensor = tf.get_default_graph().get_tensor_by_name("rnet/prob1:0")
onet_conv6_2_conv6_2_output_tensor = tf.get_default_graph().get_tensor_by_name("onet/conv6-2/conv6-2:0")
onet_conv6_3_conv6_3_output_tensor = tf.get_default_graph().get_tensor_by_name("onet/conv6-3/conv6-3:0")
onet_prob1_output_tensor = tf.get_default_graph().get_tensor_by_name("onet/prob1:0")
"""
tf.saved_model.utils.build_tensor_info的作用是构建一个TensorInfo protocol
输入参数是张量的名称、类型大小，这里是string，应是名称；输出是基于提供参数的a tensor protocol buffer
"""
pnet_inputs = tf.saved_model.utils.build_tensor_info(pnet_input_tensor)
rnet_inputs = tf.saved_model.utils.build_tensor_info(rnet_input_tensor)
onet_inputs = tf.saved_model.utils.build_tensor_info(onet_input_tensor)
pnet_conv4_2_BiasAdd_outputs = tf.saved_model.utils.build_tensor_info(pnet_conv4_2_BiasAdd_output_tensor)
pnet_prob1_outputs = tf.saved_model.utils.build_tensor_info(pnet_prob1_output_tensor)
rnet_conv5_2_conv5_2_outputs = tf.saved_model.utils.build_tensor_info(rnet_conv5_2_conv5_2_output_tensor)
rnet_prob1_outputs = tf.saved_model.utils.build_tensor_info(rnet_prob1_output_tensor)
onet_conv6_2_conv6_2_outputs = tf.saved_model.utils.build_tensor_info(onet_conv6_2_conv6_2_output_tensor)
onet_conv6_3_conv6_3_outputs = tf.saved_model.utils.build_tensor_info(onet_conv6_3_conv6_3_output_tensor)
onet_prob1_outputs = tf.saved_model.utils.build_tensor_info(onet_prob1_output_tensor)

# 整合所有的输入与输出，都是字典，key可以自定义，value是之前build_tensor_info的一个张量协议缓冲区
inputs = {
    "pnet_inputs": pnet_inputs,
    "rnet_inputs": rnet_inputs,
    "onet_inputs": onet_inputs
}
outputs = {
    "pnet_conv4_2_BiasAdd_outputs": pnet_conv4_2_BiasAdd_outputs,
    "pnet_prob1_outputs": pnet_prob1_outputs,
    "rnet_conv5_2_conv5_2_outputs": rnet_conv5_2_conv5_2_outputs,
    "rnet_prob1_outputs": rnet_prob1_outputs,
    "onet_conv6_2_conv6_2_outputs": onet_conv6_2_conv6_2_outputs,
    "onet_conv6_3_conv6_3_outputs": onet_conv6_3_conv6_3_outputs,
    "onet_prob1_outputs": onet_prob1_outputs
}

# 构建signature_def_map
signature = tf.saved_model.signature_def_utils.build_signature_def(
    inputs=inputs, outputs=outputs, method_name="mtcnn_signature"
)
"""
add_meta_graph_and_variables方法导入graph的信息及变量，该方法假设变量都已经初始化好了，
对应每个SavedModelBuilder这个方法一定要执行一次用于导入第一个meta_graph,
第一个参数：传入当前的session，包含了graph的结构与所有变量；第二个参数：给当前需要保存的
meta_graph一个标签，标签名可以自定义，在之后模型载入的时候，需要根据这个签名去查找对应的
MetaGraphDef，标签也可以选用系统定义好的参数，如：tf.saved_model.tag_constants.SERVING
与tf.saved_model.tag_constants.TRAINING
"""
builder.add_meta_graph_and_variables(sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
                                     signature_def_map={"mtcnn_signature": signature})

"""
save方法就是模型序列化到指定目录底下,保存好以后到saved_model_dir目录下，
会有一个saved_model.pb文件以及variable文件夹，variable文件夹保存所有变量，
saved_model.pb文件用于保存模型结构等信息
"""
builder.save()
sess.close()
