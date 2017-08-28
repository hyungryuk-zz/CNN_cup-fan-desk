'''
Created on 2016. 9. 28.

@author: mythrith
'''
# 라이브러리 및 데이터를 불러옵니다.
import tensorflow as tf
import re
import scipy.misc as smi
import numpy as np
import cv2

def weight_variable(shape,name):    
    initial = tf.truncated_normal(shape, stddev=0.1)    
    return tf.Variable(initial, name=name)

def bias_variable(shape,name):    
    initial = tf.constant(0.1, shape=shape)    
    return tf.Variable(initial, name=name)

def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def random_batch(arr_x,arr_y):
    num_images = len(arr_x)
    idx = np.random.choice(num_images, size=100, replace=False)
    x_batch = arr_x[idx]
    y_batch = arr_y[idx]
    return x_batch, y_batch

images_train = []    
labels_train = []
i=0
for line in open("C:/Users/ryuk/.spyder-py3/windcup.csv",'r'):
    cols = re.split(',|\n',line)
    #plt.imshow(scipy.misc.imread(cols[0]+".jpg"))
    #e = smi.imread(cols[0]+".jpg")
    images_train.append(smi.imresize(smi.imread(cols[0]+".jpg"),[28,28,3]))
    # 3rd column is label and needs to be converted to int type
    if int(cols[2])==0:
        labels_train.append(1)
        labels_train.append(0)
        labels_train.append(0)
    elif int(cols[2])==1:
        labels_train.append(0)
        labels_train.append(1)
        labels_train.append(0)
    else:
        labels_train.append(0)
        labels_train.append(0)
        labels_train.append(1)
    print(i)
    i=i+1
#    
#import matplotlib.pyplot as plt
#plt.imshow(images_train[100])  
#ex =  images_train[0].shape
images_train = np.reshape(images_train,[-1,28,28,3])
print(images_train.shape)
labels_train = np.reshape(labels_train,[-1,3])
print(labels_train.shape)
x = tf.placeholder("float",shape=[None,28,28,3])
y = tf.placeholder('float',shape=[None,3])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
L1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)

L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



# L2 Conv shape=(?, 14, 14, 64)
#    Pool     ->(?, 7, 7, 64)
# W2 의 [3, 3, 32, 64] 에서 32 는 L1 에서 출력된 W1 의 마지막 차원, 필터의 크기 입니다.
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# L2 = tf.nn.dropout(L2, keep_prob)


# FC 레이어: 입력값 7x7x64 -> 출력값 256
# Full connect를 위해 직전의 Pool 사이즈인 (?, 7, 7, 64) 를 참고하여 차원을 줄여줍니다.
#    Reshape  ->(?, 256)
W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)


# 최종 출력값 L3 에서의 출력 256개를 입력값으로 받아서 0~9 레이블인 10개의 출력값을 만듭니다.
W4 = tf.Variable(tf.random_normal([256, 3], stddev=0.01))
model = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
# 최적화 함수를 RMSPropOptimizer 로 바꿔서 결과를 확인해봅시다.
# optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)


#########
# 신경망 모델 학습
######
init = tf.initialize_all_variables()
sess = tf.Session()
#sess.close()
sess.run(init)

batch_size = 100
total_batch = int(2000 / batch_size)

for epoch in range(100):
    total_cost = 0
    print(epoch)
    for i in range(total_batch):
        
        
        batch_xs, batch_ys = random_batch(images_train,labels_train)
        print(batch_ys.shape)
        # 이미지 데이터를 CNN 모델을 위한 자료형태인 [28 28 1] 의 형태로 재구성합니다.
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={x: batch_xs,
                                          y: batch_ys,
                                          keep_prob: 0.7})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

#########
# 결과 확인
#######
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
#img = smi.imresize(smi.imread("C:/Users/ryuk/.spyder-py3/images/wind/wind634.jpg"),[28,28,3])
#img_reshape = np.reshape(img,[1,28,28,3])
#
#label=[]
#label.append(1)
#label.append(0)
#label = np.reshape(label,[1,2])
#print('정확도:', sess.run(accuracy,
#                           feed_dict={x: img_reshape,
#                                   y: label,
#                                   keep_prob: 1}))

cap = cv2.VideoCapture(0)
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
  
count = 0
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
      
    # Display the resulting frame
    cv2.imshow('Frame',frame)
 
    cv2.imwrite("frame.jpg", frame)
    img = smi.imresize(smi.imread("frame.jpg"),[28,28,3])
    img_reshape = np.reshape(img,[1,28,28,3])
    label=[]
    label.append(1)
    label.append(0)
    label.append(0)
    label = np.reshape(label,[1,3])
    
    label1=[]
    label1.append(0)
    label1.append(1)
    label1.append(0)
    label1 = np.reshape(label,[1,3])
    
    label2=[]
    label2.append(0)
    label2.append(0)
    label2.append(1)
    label2 = np.reshape(label,[1,3])
    if sess.run(accuracy,feed_dict={x: img_reshape,y: label,keep_prob: 1}) == 1.0:
        print("선풍기입니다")
    elif sess.run(accuracy,feed_dict={x: img_reshape,y: label1,keep_prob: 1}) == 1.0:
        print("종이컵입니다")
    else:
        print("종이컵이나 선풍기를 보여주세요")
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
