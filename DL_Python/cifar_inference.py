from openvino.inference_engine import IENetwork, IEPlugin
import cv2
import numpy as np

plugin = IEPlugin("CPU", "/opt/intel/openvino_2019.1.094/deployment_tools/inference_engine/lib/intel64")

model_xml = '/home/ai/work/caffe/examples/cifar10/cifar.xml'
model_bin = '/home/ai/work/caffe/examples/cifar10/cifar.bin'
print('Loading network files:\n\t{}\n\t{}'.format(model_xml, model_bin))

net = IENetwork(model=model_xml, weights=model_bin)

supported_layers = plugin.get_supported_layers(net)
not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
if len(not_supported_layers) != 0:
	print("Following layers are not supported by the plugin for specified device {}:\n {}".formaT(plugin.device, ', '.join(not_supported_layers)))
	print("Please try to specify cpu extensions library path in sample's command line parameters using –l or --cpu_extension command line argument")
	sys.exit(1)

net.batch_size = 1

input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

exec_net = plugin.load(network=net)

img = cv2.imread('/home/ai/work/caffe/data/hands/paper/2017-03-30 07.05.36.jpg', cv2.IMREAD_COLOR)

height, width, _ = img.shape
n, c, h, w = net.inputs[input_blob].shape
img2 = img
if height != h or width != w:
	img2 = cv2.resize(img, (w, h))

img2 = img2.transpose((2, 0, 1))  
# Change data layout from HWC to CHW

images = np.ndarray(shape=(n, c, h, w))
images[0] = img2

res = exec_net.infer(inputs={input_blob: images})
probs = res[out_blob]

print('Top 3 results:')
top_ind = np.argsort(probs)[0][:-4:-1]
for id in top_ind:
	print('label #{} : {:0.2f}'.format(id, probs[0][id]))

del net