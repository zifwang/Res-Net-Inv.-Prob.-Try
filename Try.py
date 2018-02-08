import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# Hyper_perameters
learning_rate = 0.001
X_dim = 20
Y_dim = X_dim
m = 10
Hidden_dim1 = 20
Hidden_dim2 = 20
Sparsity = 2
train_size = 5000
test_size = 500
lowl = 0
highl = 1

# Generate the training & testing data
# Training Data
    # Inverse problem
    # A is a martix contains random number
    # Mask is to make sure data is sparse
    # X is output
    # Y is input
# A and A_tran will use in res-net
A = np.float32(np.random.normal(size=[m,X_dim]))
print ("A size: ", A.shape)  #Find A size
#print ("A is: ", A)         #Display A number
A_tran = A.transpose()
print ("A_tran size: ", A_tran.shape)  #Find A_tran size
W_A = np.matmul(A_tran,A)
print ("W_A size: ", W_A.shape)  #Find W_A size

# Create a mask 
Mask = np.zeros(shape=[X_dim,train_size],dtype='float32')
for i in range(train_size):
    for j in random.sample(range(X_dim),Sparsity):
        Mask[:,i][j]=1
print ("Mask size: ", Mask.shape)  #Find Mask size
#print ("Mask is: ", Mask)   #Display Mask number

# Generate input and output for training Neural Network
# X_train is output
X_train = Mask*np.float32(np.random.uniform(low=lowl, high=highl, size=[X_dim,train_size]))
print ("X_train size: ", X_train.shape)  #Find X_train size
# Y_train is input 
Y_train = np.matmul(np.transpose(A),np.matmul(A,X_train))
print ("Y_train size: ", Y_train.shape)  #Find Y_train size
#print ("X_train: ",X_train) #Display X_train number
#print ("Y_train: ",Y_train) #Display Y_train number

# Testing Data
# Create a mask 
Mask = np.zeros(shape=[X_dim,test_size],dtype='float32')
for i in range(test_size):
    for j in random.sample(range(X_dim),Sparsity):
        Mask[:,i][j]=1
#print ("Mask is: ", Mask)   #Display Mask number
# Generate input and output for testing Neural Network
# X_test is output
X_test = Mask*np.float32(np.random.uniform(low=lowl, high=highl, size=[X_dim,test_size]))
# Y_test is input 
Y_test = np.matmul(np.transpose(A),np.matmul(A,X_test))
#print ("X_test: ",X_test) #Display X_test number
#print ("Y_test: ",Y_test) #Display Y_test number
###################################################################

# Initialize the Neural Network
# 3 Hidden layers
# Input size 20
Input = tf.placeholder(tf.float32, [Y_dim,None])
# print("Input size: ", Input.shape) #Find input size
# Output size 20                 
Output = tf.placeholder(tf.float32, [X_dim,None])
# print("Output size: ", Output.shape) #Find output size
# print("Input: ", Input)
# print("Output: ", Output)

# Res-net neural network
# First hidden layer using relu
W1 = tf.Variable(tf.random_normal([Hidden_dim1,Y_dim],stddev=0.35))
print("W1 size: ", W1.shape) #Find W1 size
b1 = tf.Variable(tf.random_normal([Hidden_dim1,1],stddev=0.35))
print("b1 size: ", b1.shape) #Find b1 size
y1 = tf.nn.relu(tf.matmul(W1,Input) + b1)
print("y1 size: ", y1.shape) #Find y1 size

# Second hidden layer using relu
# y2 = relu(w2*y1+b2+y1)
W2 = tf.Variable(tf.random_normal([Hidden_dim2,Hidden_dim1],stddev=0.35))
print("W2 size: ", W2.shape) #Find W2 size
b2 = tf.Variable(tf.random_normal([Hidden_dim2,1],stddev=0.35))
print("b2 size: ", b2.shape) #Find b2 size
# Before doing second layer relu, Do y2 = W2*y1+b2+(W2_R*y1+b2_R)
W2_R = tf.Variable(tf.random_normal([Hidden_dim2,Hidden_dim1],stddev=0.35))
print("W2 size: ", W2_R.shape) #Find W2_R size
b2_R = tf.Variable(tf.random_normal([Hidden_dim2,1],stddev=0.35))
print("b2 size: ", b2_R.shape) #Find b2_R size
y2 = tf.matmul(W2,y1) + b2  #+ tf.matmul(W_A,y1) + b2_R 
y2 = tf.nn.relu(y2)
print("y2 size: ", y2.shape) #Find y2 size

# Third hidden layer using relu
W3 = tf.Variable(tf.random_normal([X_dim,Hidden_dim2],stddev=0.35))
print("W3 size: ", W3.shape) #Find W3 size
b3 = tf.Variable(tf.random_normal([X_dim,1],stddev=0.35))
print("b3 size: ", b3.shape) #Find b3 size
# Before doing second layer relu, Do y3 = W3*y2+b3+(W3_R*y2+b3_R)
W3_R = tf.Variable(tf.random_normal([X_dim,Hidden_dim2],stddev=0.35))
print("W3_R size: ", W3_R.shape) #Find W3_R size
b3_R = tf.Variable(tf.random_normal([X_dim,1],stddev=0.35))
print("b3_R size: ", b3_R.shape) #Find b3_R size
# (I-alpha*A_tran*A)*y2
# Generate a random number
alpha = random.randint(1,10)*0.1
print (alpha)
# Generate a identity matrix
I = np.identity(20)
print ("I size: ", I.shape) #Find I size
W3_A = (I - alpha*W_A)
W3_A = np.array(W3_A,dtype=np.float32)
print (W3_A.dtype)
print ("W3_A size: ", W3_A.shape)
print ("W3_R: ", W3_R)
print ("W3: ", W3)
y3 = tf.matmul(W3,y2) + b3 + tf.matmul(W3_A,y1)#tf.matmul(W3_A,y1)
y3 = tf.nn.relu(y3)
print("y3 size: ", y3.shape) #Find y3 size

# Calculate loss by l2 nomral
y = tf.subtract(y3,Output)
loss = tf.reduce_mean(tf.nn.l2_loss(y)) # + tf.norm(W1) + tf.norm(W2) + tf.norm(W3) + tf.norm(W4)

# Use adamoptimizion to train
lr = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# Initialize tensor flow
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Creat record base
train_loss_record=[]
test_loss_record=[]
#################################################################

# Start Training
for i in range(100000):#
      
  _,loss_train=sess.run([train_step,loss], feed_dict={lr:learning_rate*(0.99**(i//100)) ,Input:Y_train, Output:X_train}) 
  train_loss_record.append(loss_train)
  
  if i%1000==0:
      #print(loss_train)
      print('The %i iteration loss: %f'%(i,loss_train))
      test_loss_record.append(sess.run(loss, feed_dict={Input:Y_train, Output:X_train}))
##################################################################

plt.figure(1)
plt.plot(train_loss_record)
plt.show()
plt.ylabel('train loss')
plt.xlabel('iterations ')
print('Train loss: %f'%train_loss_record[-1])

plt.figure(2)
plt.plot(test_loss_record)
plt.ylabel('test loss')
plt.xlabel('iterations ')
print('Test loss: %f'%test_loss_record[-1])
#print('Done')

#%%
x_sample = Mask[:,0:1]*np.float32(np.random.uniform(low=lowl, high=highl, size=[X_dim,1]))
y_sample = np.matmul(np.transpose(A),np.matmul(A,x_sample))
x_output = sess.run(y3,feed_dict={Input:y_sample})
print('True output:')
print(x_sample)
print('Predicted output:')
print(x_output)
print('The diff norm: %f'%np.linalg.norm(x_output-x_sample))
print('The original norm: %f'%np.linalg.norm(x_sample))
print(sum(sum(x_output>0)))
plt.plot(x_sample)
plt.plot(x_output)
