#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

CXX = c++
CXXFLAGS = -pthread -std=c++0x
OBJS = args.o dictionary.o matrix.o qmatrix.o vector.o model.o utils.o fasttext.o fasttext_wrapper.o productquantizer.o
INCLUDES = -I.

opt: CXXFLAGS += -O3 -funroll-loops
opt: build

debug: CXXFLAGS += -g -O0 -fno-inline
debug: fasttext

args.o: src/args.cc src/args.h
	$(CXX) $(CXXFLAGS) -c src/args.cc

dictionary.o: src/dictionary.cc src/dictionary.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/dictionary.cc

matrix.o: src/matrix.cc src/matrix.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/matrix.cc
	
qmatrix.o: src/qmatrix.cc src/qmatrix.h
	$(CXX) $(CXXFLAGS) -c src/qmatrix.cc

vector.o: src/vector.cc src/vector.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/vector.cc

model.o: src/model.cc src/model.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/model.cc

productquantizer.o: src/productquantizer.cc src/productquantizer.h
	$(CXX) $(CXXFLAGS) -c src/productquantizer.cc

utils.o: src/utils.cc src/utils.h
	$(CXX) $(CXXFLAGS) -c src/utils.cc

fasttext.o: src/fasttext.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/fasttext.cc

fasttext_wrapper.o: src/fasttext_wrapper.cc src/fasttext.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/fasttext_wrapper.cc

libfasttext.a: $(OBJS)
	$(AR) rcs libfasttext.a $(OBJS)

clean:
	rm -rf *.o libfasttext.a

build: libfasttext.a
	go build

test: build
	go test
