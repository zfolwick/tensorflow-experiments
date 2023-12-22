# tensor tutorial


import tensorflow as tf

x = tf.constant(5)
print(x)

# y = tf.constant([1,2,3]) #rank-1
# print(y)

# z = tf.constant([[1,2,3],[4,5,6]])
# print(z[0,1:3])
#reshaping


# ones = tf.ones((3,3))
# print(ones)

# zeros = tf.zeros((3,3))
# print(zeros)

# eye = tf.eye(3)
# print(eye)

# rnd = tf.random.normal((3,3), mean=0, stddev=1)
# print(rnd)

# uniform = tf.random.uniform((3,3), minval=0, maxval=1)
# print(uniform)

# rng = tf.range(10)
# print(rng)

# #cast
# x = tf.cast(x, tf.float32)
# print(x)

#elementwise
# first = tf.constant([1,2,3])
# second = tf.constant([4,5,6])
# third = tf.tensordot(first, second, axes=1)
# print(third)

# # matrix multiplation
# left = tf.random.normal((2,3))
# right = tf.random.normal((3,4))
# mult = tf.matmul(left, right)
# print(mult)
# print("again!")
# print(left @ right)

# rshap = tf.random.uniform((2,3))
# print(rshap)

# newshape = tf.reshape(rshap, (6)) # (3,2)
# print(newshape)

# #numpy
# nrml = tf.random.normal((2,3))
# nrml = nrml.numpy()
# print(nrml)
# print(type(nrml))

# nrml = tf.convert_to_tensor(nrml)
# print(type(nrml))

#string tensor
str = tf.constant("derp")
print(str)

strarray = tf.constant(["derp", "herp", "flerp"])
print(strarray)

#variable
v = tf.Variable([1,2,3]) # keras will be doing all this for us.
print(v)