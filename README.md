##Main code is copy from https://github.com/pjreddie/darknet##

#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

##Below is my work:##

##Save detected human positions and other information in a csv file##
```
./darknet detector humansave cfg/coco.data cfg/yolo.cfg yolo.weights data/20170621180731.avi
```

##Modified functions and added files##
   	Modified functions
	```
	1. examples/detector.c run_detector add 697-705 humansave part
        2. Makefile add humansave.o
	```
	Add files
	```
	1. humansave.c
	2. humansave.h
	```

