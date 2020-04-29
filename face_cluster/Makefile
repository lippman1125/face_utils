CROSS_COMPILE ?=

CC = $(CROSS_COMPILE)gcc
CXX = $(CROSS_COMPILE)g++
LD = $(CROSS_COMPILE)ld
AR = $(CROSS_COMPILE)ar cr
STRIP = $(CROSS_COMPILE)strip

# GIT_VERSION=$(shell git show -s --pretty=format:%h)
COMPILE_DATE=$(shell date +"%Y-%m-%d %H:%M:%S")

MACRO_DEFS += -DGIT_VERSION="\"$(GIT_VERSION)\"" \
               -DCOMPILE_DATE="\"$(COMPILE_DATE)\""\
               -DGPU_ONLY

CAFFE_INSTALL_DIR=/home/lqy/workshop/caffe
CUDA_INSTALL_DIR=/usr/local/cuda
INCLUDES = -I./\

CXXFLAGS ?= $(INCLUDES)

CXXFLAGS += -Wall \
            -Wno-unknown-pragmas \
            -fPIC \
            -fexceptions \
            -O3 \
            -std=c++11 \

CXXFLAGS += $(MACRO_DEFS) \


LDFLAGS += -ldlib


SRC_PATH = ./
DIRS = $(shell find $(SRC_PATH) -maxdepth 3 -type d)
SRCS = $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cpp))
OBJS = $(patsubst %.cpp, %.o, $(SRCS))

LIB = libface_cluster.so
TARGET = face_cluster
all: $(TARGET) $(LIB)

face_cluster:$(TARGET)
$(TARGET):$(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)
	$(STRIP) $@


lib: $(LIB)
$(LIB):cluster_imp.o cluster.o math_utils.o
	$(CXX) -shared -o $@ $^

%.o:%.cpp
	@echo $(SRCS)
	@echo $(OBJS)
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -f $(OBJS)
distclean: clean
	rm -f $(TARGET)
	rm -f $(LIB)

