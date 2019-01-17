import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])

matrix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matrix1, matrix2)

sess = tf.Session()
result = sess.run(product)
print (result)
sess.close()


with tf.Session() as sess:
  result = sess.run([product])
  print (result)

# Tensor 是张量， shape 是维度， dtype 是数据类型 ，tf.Variable 是变量
data1 = tf.constant(2,dtype = tf.int32)
print(data1)
sess = tf.Session()
print(sess.run(data1))
sess.close()

data2 = tf.Variable(10,name='var')
print(data2)
sess = tf.Session()
init = tf.global_variables_initializer()
# 如果不初始化会报错
sess.run(init)
print(sess.run(data2))

# 常量四则运算
data3 = tf.constant(6)
data4 = tf.constant(2)
dataAdd = tf.add(data3,data4) # 加
dataSub = tf.subtract(data3,data4) # 减
dataMul = tf.multiply(data3,data4) # 乘
dataDiv = tf.divide(data3,data4) # 除
with tf.Session() as sess:
    print (sess.run(dataAdd))
    print (sess.run(dataSub))
    print (sess.run(dataMul))
    print (sess.run(dataDiv))
print('End!')

# 变量四则运算
data5 = tf.constant(6)
data6 = tf.Variable(4)
dataAdd = tf.add(data5,data6) # 加
dataCopy = tf.assign(data6,dataAdd)
# 把dataAdd的运算结果，赋值到data6
dataSub = tf.subtract(data5,data6) # 减
dataMul = tf.multiply(data5,data6) # 乘
dataDiv = tf.divide(data5,data6) # 除

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print ('加：',sess.run(dataAdd))
    print ('减：',sess.run(dataSub))
    print ('乘：',sess.run(dataMul))
    print ('除：',sess.run(dataDiv))
    print ('dataCopy :',sess.run(dataCopy))
    # dataAdd  = 10
    print ('dataCopy.eval() :',dataCopy.eval())
    # eval(expression[, globals[, locals]]) , 用来执行一个字符串表达式，并返回表达式的值
    # dataAdd + data5 = 10 + 6 = 16
    print ('tf.get_default_session() :',tf.get_default_session().run(dataCopy))
    # dataCopy + data5 = 16 + 6 = 22
print('End!')
