# HOW TO USE:
# For training: 
# $ make
# For testing:
# $ make -e runtype=test
CXX:= g++ -std=c++11
CFLAGS:= -fpermissive -fopenmp -O2
SRC:= src/*.cpp src/tinyxml2/*.cpp
mainfunc:= demo/rf_train_commandline.cpp
target:= rf_train 
runtype:= train
ifeq ($(runtype),test)
	target:= rf_test
        mainfunc:= demo/rf_test_commandline.cpp
endif
$(target):$(SRC)
	$(CXX) $(CFLAGS) $^ $(mainfunc) -o $@

clean:
	$(RM) $(target)
