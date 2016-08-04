# -*- coding: utf-8 -*-
"""
@brief: 人脸归一化对齐处理工具

@author: Riwei.Chen <riwei.chen@outlook.com>

@Todo: 有时候无法计算有效的仿射变换矩阵。
"""
import os
import cv2
import dlib
import numpy as np
import skimage
import skimage.io

# this template is reference from 
TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)


class FaceAligner:
    """
    @brief: 人脸检测、对齐的类工具
    """

    #: Landmark indices corresponding to the inner eyes and bottom lip.
    INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]

    #: Landmark indices corresponding to the outer eyes and nose.
    OUTER_EYES_AND_NOSE = [36, 45, 33]

    def __init__(self, facePredictor, padding=0.2):
        """
        """
        assert facePredictor is not None

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(facePredictor)
        self.padding = padding
        # add padding
        new_template =[]
        for item in  TEMPLATE:
            new_item = ((item[0]+self.padding)/(2*self.padding+1),(item[1]+self.padding)/(2*self.padding+1))
            new_template.append(new_item)
        self.new_template = np.float32(new_template)
        
        # this shape is reference from dlib implement.
        self.mean_shape_x = [0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
            0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
            0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
            0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
            0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
            0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
            0.553364, 0.490127, 0.42689]#17-67
        self.mean_shape_y = [0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
            0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
            0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
            0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
            0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
            0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
            0.784792, 0.824182, 0.831803, 0.824182] #17-67

    def getAllFaceBoundingBoxes(self, rgbImg):
        """
        @brief 返回所有的人脸框bbox
        """
        assert rgbImg is not None
        try:
            return self.detector(rgbImg, 1)
        except Exception as e:
            print("Warning: {}".format(e))
            return []

    def getLargestFaceBoundingBox(self, rgbImg):
        """
        @brief 返回最大的人脸框。根据 w*h
        """
        assert rgbImg is not None
        faces = self.getAllFaceBoundingBoxes(rgbImg)
        if len(faces) > 0:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None

    def findLandmarks(self, rgbImg, bb):
        """
        @brief 返回人脸的关键点信息
        @return type:list of (x,y) 
        """
        assert rgbImg is not None
        assert bb is not None
        points = self.predictor(rgbImg, bb)
        return list(map(lambda p: (p.x, p.y), points.parts()))

    def align_openface(self, imgDim, rgbImg, bb=None,
              landmarks=None, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP):
        '''
        @brief 与Openface中的实现相同，不足的地方在于裁剪的效果并不好。
        '''
        assert imgDim is not None
        assert rgbImg is not None
        assert landmarkIndices is not None
        if bb is None:
            bb = self.getLargestFaceBoundingBox(rgbImg)
            if bb is None:
                return
        if landmarks is None:
            landmarks = self.findLandmarks(rgbImg, bb)

        npLandmarks = np.float32(landmarks)
        npLandmarkIndices = np.array(landmarkIndices)
        H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],
                                   imgDim * MINMAX_TEMPLATE[npLandmarkIndices])
        aligned_face = cv2.warpAffine(rgbImg, H, (imgDim, imgDim))
        return aligned_face
    
    def align_similarity(self, imgDim, rgbImg, bb=None,
              landmarks=None, landmarkIndices=range(0,68)):
        '''
        @brief 与align_openface()类似，但是通过更多个的点来计算仿射变换矩阵。
        '''
        assert imgDim is not None
        assert rgbImg is not None
        assert landmarkIndices is not None
        if bb is None:
            bb = self.getLargestFaceBoundingBox(rgbImg)
            if bb is None:
                return
        if landmarks is None:
            landmarks = self.findLandmarks(rgbImg, bb)
        npLandmarkIndices = np.array(landmarkIndices)
        npLandmarks = np.array(landmarks).astype(np.int)
        T = (imgDim * self.new_template).astype(np.int)
        source = np.reshape(npLandmarks[npLandmarkIndices],(1,68,2))
        target = np.reshape(T[npLandmarkIndices],(1,68,2))
        H = cv2.estimateRigidTransform(source,target,False)
        if H is None:
            return None
        else:
            aligned_face = cv2.warpAffine(rgbImg, H, (imgDim, imgDim))
            return aligned_face

    def align_dlib_cpp(self, imgDim, rgbImg, bb=None):
        '''
        @brief: 与dlib C++版本实现的裁剪对齐方法一致。

        @attention 
        '''
        assert imgDim is not None
        assert rgbImg is not None
        if bb is None:
            bb = self.getLargestFaceBoundingBox(rgbImg)
            if bb is None:
                return
        landmarks = self.findLandmarks(rgbImg, bb)
        shape_x = [landmarks[i][0] for i in range(68)]
        shape_y = [landmarks[i][1] for i in range(68)]

        from_points = []
        to_points = []
        
        for i in range(17,68):
            # 忽略掉低于嘴唇的部分
            if i>=55 and i<=59:
                continue
            # 忽略眉毛部分
            if i >=17 and i<=26:
                continue
            # 上下左右都padding
            new_ref_x = (self.padding+self.mean_shape_x[i-17])/(2*self.padding+1)
            new_ref_y = (self.padding+self.mean_shape_y[i-17])/(2*self.padding+1)

            from_points.append((shape_x[i],shape_y[i]))
            to_points.append((imgDim *new_ref_x,imgDim *new_ref_y))
            
        source = np.array(from_points).astype(np.int)
        target = np.array(to_points,).astype(np.int)
        source = np.reshape(source,(1,36,2))
        target = np.reshape(target,(1,36,2))

        H = cv2.estimateRigidTransform(source,target,False)
        if H is None:
            return
        else:
            aligned_face = cv2.warpAffine(rgbImg, H, (imgDim, imgDim))
            return aligned_face

        
    def align_5_points(self, imgDim, rgbImg, five_points):
        '''
        @brief 根据：两个眼角，两个嘴角，一个鼻子这五点进行对齐,其它与align_dlib_cpp保持一致。

        @param five_points: 一个shape 为10 的人脸关键点坐标位置。
        @attention 这个版本适合于两个眼角和鼻子，嘴角
        
        '''
        assert imgDim is not None
        assert rgbImg is not None
        from_points = []
        for i in range(0,10,2):
            from_points.append((five_points[i],five_points[i+1]))

        to_points = []
        for i in [36-17,45-17,33-17,48-17,54-17]:
            new_ref_x = (self.padding+self.mean_shape_x[i])/(2*self.padding+1)
            new_ref_y = (self.padding+self.mean_shape_y[i])/(2*self.padding+1)
            to_points.append((imgDim *new_ref_x,imgDim *new_ref_y))
            
        source = np.array(from_points).astype(np.int)
        target = np.array(to_points,).astype(np.int)
        source = np.reshape(source,(1,5,2))
        target = np.reshape(target,(1,5,2))

        H = cv2.estimateRigidTransform(source,target,False)
        if H is None:
            return
        else:
            aligned_face = cv2.warpAffine(rgbImg, H, (imgDim, imgDim))
            return aligned_face
        
    def align_5_points_eye_center(self, imgDim, rgbImg, five_points):
        '''
        @brief 根据：两个眼睛中心，两个嘴角，一个鼻子这五点进行对齐,其它与align_dlib_cpp保持一致。

        @param five_points: 一个shape 为10 的人脸关键点坐标位置。
        @attention 适合于CelebA提供的数据。
        
        '''
        assert imgDim is not None
        assert rgbImg is not None
        from_points = []
        for i in range(0,10,2):
            from_points.append((five_points[i],five_points[i+1]))

        to_points = []
        five_mean_x = [(self.mean_shape_x[36-17]+self.mean_shape_x[39-17])*0.5,(self.mean_shape_x[42-17]+self.mean_shape_x[45-17])*0.5,self.mean_shape_x[33-17],self.mean_shape_x[48-17],self.mean_shape_x[54-17]]
        five_mean_y = [(self.mean_shape_y[36-17]+self.mean_shape_y[39-17])*0.5,(self.mean_shape_y[42-17]+self.mean_shape_y[45-17])*0.5,self.mean_shape_y[33-17],self.mean_shape_y[48-17],self.mean_shape_y[54-17]]
        for i in range(5):
            new_ref_x = (self.padding+five_mean_x[i])/(2*self.padding+1)
            new_ref_y = (self.padding+five_mean_y[i])/(2*self.padding+1)
            to_points.append((imgDim *new_ref_x,imgDim *new_ref_y))
            
        source = np.array(from_points).astype(np.int)
        target = np.array(to_points,).astype(np.int)
        source = np.reshape(source,(1,5,2))
        target = np.reshape(target,(1,5,2))

        H = cv2.estimateRigidTransform(source,target,False)
        if H is None:
            return
        else:
            aligned_face = cv2.warpAffine(rgbImg, H, (imgDim, imgDim))
            return aligned_face

    def get_normalized_faces(self,file_name):
        """
        @brief get multi_face from file.
        """
        img =skimage.io.imread(file_name)
        (ret_bbs,ret_faces) = self.get_normalized_faces_from_image(img)
        return (ret_bbs,ret_faces)
       
    def get_normalized_faces_from_image(self,img):
        '''
        @brief get multi_face from a img
        '''
        bbs = self.getAllFaceBoundingBoxes(img)
        ret_bbs = []
        ret_faces = []
        if bbs is not []:
            for bb in bbs:
                face = self.align_dlib_cpp(imgDim, img, bb)
                if face is not None:
                    ret_bbs.append((bb.left(),bb.right(),bb.top(),bb.bottom()))
                    ret_faces.append(face)
        return (ret_bbs,ret_faces)  	

    def align_multi_face_and_save(self,file_path,save_path):
        '''
        @brief 一个文件夹进行多个人脸的检测与对齐
        '''
        file_list = os.listdir(file_path)
        imgDim = 128
        for file_name in file_list:
            print "processing ",file_name
            bgrImg = cv2.imread(file_path+file_name)
            rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
            bbs = self.getAllFaceBoundingBoxes(rgbImg)
            if bbs is not []:
                for bb in bbs:
                    face = self.align_dlib_cpp(imgDim, rgbImg, bb)
                    if face is not None:
                        rgbImg_face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_path+file_name+"@"+str(bb.left())+"_"+str(bb.top())+"_"+str(bb.right())+"_"+str(bb.bottom())+".jpg",rgbImg_face)

              
if __name__ =="__main__":
    facePredictor = "data/shape_predictor_68_face_landmarks.dat"
    align = FaceAligner(facePredictor,0.2)
    imgDim = 128
    file_name = "test.jpg"
    print FaceAligner.get_normalized_faces(file_name)
    

    
