# DeepFace
基于开源框架实现的人脸识别、脸脸检测、人脸关键点检测等任务
各个任务分别在FaceDetection, FaceAlignment, FaceRecognition 三个文件中

##人脸检测
  baseline: 基于基于滑动窗口的人脸检测，将训练好了的网络改为全卷积网络，然后利用全卷积网络对于任意大小的图像输入，进行获取输出HeapMap。
  
##人脸关键点检测
  try1_1： 基于DeepID网络结构的人脸关键点检测

##人脸验证
  deepid： 基于DeepID网络结构的人脸验证




欢迎大家提出改进意见。
