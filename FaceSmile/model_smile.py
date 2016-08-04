#=====================================
import caffe
caffe.set_mode_cpu()
averageImg = [129.1863,104.7624,93.5940]
model_path ='deepface/model/'
model_define=model_path+'deploy_smile.prototxt'
model_weight=model_path+'face_smile.model'
feature_layer='classfy_smile'
image_formats =['jpg','png','bmp']
data_w = 128
data_h =  128
scale = 1
data_as_gray = False
sub_mean = False
net = caffe.Classifier(model_define, model_weight)
#====================================