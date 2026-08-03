set(DRIVER_SOURCES
	Device.cpp
)

# Add CUDA sources if CUDA is available
if(PROTON_ENABLE_CUDA)
	list(APPEND DRIVER_SOURCES
		GPU/CudaApi.cpp
		GPU/CuptiApi.cpp
		GPU/NvtxApi.cpp
	)
endif()

# Add HIP/ROCm sources if ROCm is available
if(PROTON_ENABLE_HIP)
	list(APPEND DRIVER_SOURCES
		GPU/HipApi.cpp
		GPU/HsaApi.cpp
		GPU/RoctracerApi.cpp
	)
endif()

# Add NPU/Ascend sources if NPU is available
if(PROTON_ENABLE_NPU)
	list(APPEND DRIVER_SOURCES
		NPU/AscendApi.cpp
		NPU/MsprofApi.cpp
	)
endif()

add_proton_library(ProtonDriver ${DRIVER_SOURCES})

# Add compile definitions
if(PROTON_ENABLE_CUDA)
	target_compile_definitions(ProtonDriver PRIVATE PROTON_ENABLE_CUDA)
endif()
if(PROTON_ENABLE_HIP)
	target_compile_definitions(ProtonDriver PRIVATE PROTON_ENABLE_HIP)
endif()
if(PROTON_ENABLE_NPU)
	target_compile_definitions(ProtonDriver PRIVATE PROTON_ENABLE_NPU)
endif()
