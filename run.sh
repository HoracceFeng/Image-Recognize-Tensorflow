sudo docker run -it $1 \
	-v /Users/horacce/Nirva/Project/Image-Recognize-Tensorflow:/code \
	-v /Users/horacce/Nirva/Data/sign_classify/DATA:/data \
	-p 60060:6006 \
	10.202.107.19/sfai/ubuntu16.04-cuda8.0-opencv3.2-cudnn6-tensorflow1.3.0_keras_tflearn:cpu bash
