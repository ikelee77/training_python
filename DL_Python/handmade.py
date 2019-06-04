import os
import cv2

original_path = '/home/ai/work/caffe/data/hands/'
part = ['final/train/', 'final/valid/']
category = ['rock/', 'paper/', 'scissors/']

params = list()
params.append(cv2.IMWRITE_PNG_COMPRESSION)
params.append(8)

for i in range(3):
	work_dir = original_path
	work_dir += category[i]

	print('data path : ' + work_dir)
	if not os.path.exists(work_dir): os.makedirs(work_dir)

for i in range(2):
	src = original_path+part[i]
	src += 'c';
	for j in range(3):
		work_dir = src + chr(ord('0')+j)
		dst = original_path+category[j]
		print('find path : ' + work_dir + ' --> ', end = '')
		if not os.path.exists(work_dir): print('path does not exist')
		else :
			filenames = os.listdir(work_dir)
			for filename in filenames:
				full_filename = os.path.join(work_dir, filename)
				if not os.path.isdir(full_filename):
					src_fn = work_dir+ '/' + filename
					dst_fn = dst + filename
					orgimg = cv2.imread(src_fn, cv2.IMREAD_COLOR)
					cutimg = cv2.resize(orgimg, (32, 32))
					cv2.imwrite(dst_fn, cutimg, params)
			print('processed')