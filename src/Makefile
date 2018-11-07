

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

LIBRARIES := ~/library/lib/libboost_program_options.a 

INCLUDES := -I ~/library/include/

GENCODE_FLAGS := -gencode arch=compute_61,code=sm_61
#-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35

OBJECTS := ParticleSystem.o main.o ParticleSystem_cuda.o World.o \
DirectChecking.o SortingContactDetection.o SortingContactDetection_cuda.o \
DirectMapping.o DirectMapping_cuda.o CellMapping.o CellMapping_cuda.o \
SortingAndSearch.o SortingAndSearch_cuda.o

main.o: main.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
        
ParticleSystem.o: ParticleSystem.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
ParticleSystem_cuda.o: ParticleSystem_cuda.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<

World.o: World.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<

DirectMapping.o: DirectMapping.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
DirectChecking.o: DirectChecking.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<
	
DirectMapping_cuda.o: DirectMapping_cuda.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<

SortingContactDetection.o: SortingContactDetection.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

SortingContactDetection_cuda.o: SortingContactDetection_cuda.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<

CellMapping.o: CellMapping.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
CellMapping_cuda.o: CellMapping_cuda.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<

SortingAndSearch.o: SortingAndSearch.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
        
SortingAndSearch_cuda.o: SortingAndSearch_cuda.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<

particles: $(OBJECTS)
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	
all: particles

test: particles
	./particles

clean:
	rm *.o
	rm particles
