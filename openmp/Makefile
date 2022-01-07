DEBUG ?= 0

RELWITHDEBINFO ?= 0

ifeq ($(strip $(COMP)),1)
CXX = clang++
else 
CXX = g++
endif

CXXFLAGS = -std=c++11 -Wall -fopenmp
LDFLAGS  = -std=c++11 -Wall -fopenmp

ifeq ($(strip $(DEBUG)),1)
CXXFLAGS += -g -O0
LDFLAGS  += -g -O0
else ifeq ($(strip $(RELWITHDEBINFO)),1)
CXXFLAGS += -O2 -g -DNDEBUG
LDFLAGS += -O2 -g -DNDEBUG
else
CXXFLAGS += -O3 -march=native
LDFLAGS  += -O3 -march=native
endif

# Include directories
INC        = -I../core
INC_EXT    =  

# Location of the libraries.
LIB        = -L../core -lcore_s
LIB_EXT    = 

INC := $(INC) $(INC_EXT)
LIB := $(LIB) $(LIB_EXT)

CXXFLAGS += $(INC)

include ../core/make_blas.mk

TARGET = main main_onetask main_buffer main_buffer2
all: $(TARGET)

.PRECIOUS: %.cc %.o

main.o: main.cc ../core/timer.h
	$(CXX) -c $(CXXFLAGS) $<

main: main.o
	$(CXX) $^ $(LIB) $(LDFLAGS) -o $@ 

main_onetask.o: main_onetask.cc ../core/timer.h
	$(CXX) -c $(CXXFLAGS) $<

main_onetask: main_onetask.o
	$(CXX) $^ $(LIB) $(LDFLAGS) -o $@ 

main_buffer.o: main_buffer.cc ../core/timer.h
	$(CXX) -c $(CXXFLAGS) $<

main_buffer: main_buffer.o
	$(CXX) $^ $(LIB) $(LDFLAGS) -o $@ 
	
main_buffer2.o: main_buffer2.cc ../core/timer.h
	$(CXX) -c $(CXXFLAGS) $<

main_buffer2: main_buffer2.o
	$(CXX) $^ $(LIB) $(LDFLAGS) -o $@ 

clean:
	rm -f *.o
	rm -f $(TARGET)

.PHONY: all clean
