CUDA_COMPILER = nvcc
CXX_COMPILER = g++

CUDA_FLAGS = -O3 -arch=sm_35
CXX_FLAGS = -O3 -std=c++11 -Wall -fopenmp

ifeq ($(PRINT_FINAL_CENTROIDS),1)
    CXX_FLAGS += -DPRINT_FINAL_CENTROIDS
	CUDA_FLAGS += -DPRINT_FINAL_CENTROIDS
endif

ifeq ($(EXPORT_FINAL_RESULT),1)
    CXX_FLAGS += -DEXPORT_FINAL_RESULT
	CUDA_FLAGS += -DEXPORT_FINAL_RESULT
endif


K_FLAG = $(if $(K),-DK=$(K),)
CXX_FLAGS += $(K_FLAG)
CUDA_FLAGS += $(K_FLAG)

BUILD_DIR = build
SRC_DIR = src

EXECUTABLE_CUDA = $(BUILD_DIR)/kmeansCuda
EXECUTABLE_CUDAV2 = $(BUILD_DIR)/kmeansCudaV2
EXECUTABLE_CUDAV3 = $(BUILD_DIR)/kmeansCudaV3
EXECUTABLE_CUDAV4 = $(BUILD_DIR)/kmeansCudaV4
EXECUTABLE_SEQ  = $(BUILD_DIR)/kmeansSequential
EXECUTABLE_PAR  = $(BUILD_DIR)/kmeansParallel

CUDA_SRC     = $(SRC_DIR)/kmeansCuda.cu
CUDAV2_SRC     = $(SRC_DIR)/kmeansCudaV2.cu
CUDAV3_SRC     = $(SRC_DIR)/kmeansCudaV3.cu
CUDAV4_SRC     = $(SRC_DIR)/kmeansCudaV4.cu
CXX_SRC      = $(SRC_DIR)/kmeansSequential.cpp
CXX_PAR_SRC  = $(SRC_DIR)/kmeansParallel.cpp

all: $(EXECUTABLE_CUDA) $(EXECUTABLE_CUDAV2) $(EXECUTABLE_CUDAV3) $(EXECUTABLE_CUDAV4) $(EXECUTABLE_SEQ) $(EXECUTABLE_PAR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(EXECUTABLE_CUDA): $(CUDA_SRC) | $(BUILD_DIR)
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CUDA_SRC) -o $(EXECUTABLE_CUDA)

$(EXECUTABLE_CUDAV2): $(CUDAV2_SRC) | $(BUILD_DIR)
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CUDAV2_SRC) -o $(EXECUTABLE_CUDAV2)

$(EXECUTABLE_CUDAV3): $(CUDAV3_SRC) | $(BUILD_DIR)
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CUDAV3_SRC) -o $(EXECUTABLE_CUDAV3)

$(EXECUTABLE_CUDAV4): $(CUDAV4_SRC) | $(BUILD_DIR)
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CUDAV4_SRC) -o $(EXECUTABLE_CUDAV4)

$(EXECUTABLE_SEQ): $(CXX_SRC) | $(BUILD_DIR)
	$(CXX_COMPILER) $(CXX_FLAGS) $(CXX_SRC) -o $(EXECUTABLE_SEQ)

$(EXECUTABLE_PAR): $(CXX_PAR_SRC) | $(BUILD_DIR)
	$(CXX_COMPILER) $(CXX_FLAGS) $(CXX_PAR_SRC) -o $(EXECUTABLE_PAR)

clean:
	rm -rf $(BUILD_DIR)
