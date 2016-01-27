

NVCC := nvcc
CC := g++

EXEC ?= @echo "[@]"

NVCCFLAGS :=

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -DDEBUG -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS := $(NVCCFLAGS)

ALL_LDFLAGS := $(ALL_CCFLAGS)

LIBRARIES :=

INCLUDES :=

GENCODE_FLAGS := -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35

OBJECTS := ParticleSystem.o main.o ParticleSystem_cuda.o

main.o: main.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
        
ParticleSystem.o: ParticleSystem.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

ParticleSystem_cuda.o: ParticleSystem_cuda.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

particles: $(OBJECTS)
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	
all: particles

clean:
	rm *.o
	rm particles
