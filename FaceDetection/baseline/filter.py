


def re_verify(net_vf, img):
    img= tf.resize(img,(face_w,face_w))
    transformer = caffe.io.Transformer({'data': net_vf.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', raw_scale)
    out = net_vf.forward_all(data=np.asarray([transformer.preprocess('data', img)]))
    return True if  out['prob'][0,map_idx] > threshold else False


net_vf = caffe.Net(model_define, model_weight, caffe.TEST)
    
