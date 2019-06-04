import cv2
import numpy as np
import os, sys

name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
name_cnt = list([0]*10)
kCIFARBatchSize = 10000
kCIFARImageNBytes = 3072
buffer = np.zeros((kCIFARImageNBytes,), np.uint8)

mat = np.zeros((32,32,3), np.uint8)
params = list()
params.append(cv2.IMWRITE_PNG_COMPRESSION)
params.append(8)

path = '/home/ai/work/caffe/data/cifar10'

for i in range(10):
	data_dir = path; 
	data_dir += '/';
	data_dir += name[i];
	print('data path : '+data_dir)
	if not os.path.exists(data_dir): os.makedirs(data_dir)

for i in range(5):
	fn = path + '/data_batch_' + str(i+1) + '.bin'
	print('filename : ' + fn)
	try: 
		data_file = open(fn, 'rb')
		for itemid in range(kCIFARBatchSize):
			label_char = data_file.read(1)
			label = ord(label_char);
			buffer = data_file.read(kCIFARImageNBytes)
			for ii in range(32 * 32):
				#opencv actually uses BGR instead of RGB
				y = int(ii/32)
				x = int(ii%32)
				mat[y,x,0] = buffer[2 * 32 * 32 + ii] #B
				mat[y,x,1] = buffer[1 * 32 * 32 + ii] #G
				mat[y,x,2] = buffer[0 * 32 * 32 + ii] #R
			filename = path + '/' + name[label] + '/' + name[label] + '%05d'%(name_cnt[label])
			filename += '.png'
			name_cnt[label] += 1
			cv2.imwrite(filename, mat, params)
		data_file.close()  
	except IOError:
		sys.exit(str(i+1) + ' of 5 training file load fail')

fn = path + '/test_batch.bin'
print('test filename : ' + fn)
try: 
	data_file = open(fn, 'rb')
	for itemid in range(kCIFARBatchSize):
		label_char = data_file.read(1)
		label = ord(label_char);
		buffer = data_file.read(kCIFARImageNBytes)
		for ii in range(32 * 32):
			#opencv actually uses BGR instead of RGB
			y = int(ii/32)
			x = int(ii%32)
			mat[y,x,0] = buffer[2 * 32 * 32 + ii] #B
			mat[y,x,1] = buffer[1 * 32 * 32 + ii] #G
			mat[y,x,2] = buffer[0 * 32 * 32 + ii] #R
		filename = path + '/' + name[label] + '/' + name[label] + '%05d'%(name_cnt[label])
		filename += '.png'
		name_cnt[label] += 1
		cv2.imwrite(filename, mat, params)
	data_file.close()  
except IOError:
	sys.exit(' test file load fail')