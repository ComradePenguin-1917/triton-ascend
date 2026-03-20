set(PROFILER_SOURCES
	Instrumentation/InstrumentationProfiler.cpp
	Instrumentation/Metadata.cpp
)

# Add CUDA sources if CUDA is available
if(PROTON_ENABLE_CUDA)
	list(APPEND PROFILER_SOURCES
		Cupti/CuptiPCSampling.cpp
		Cupti/CuptiProfiler.cpp
		Instrumentation/CudaRuntime.cpp
	)
endif()

# Add HIP/ROCm sources if ROCm is available
if(PROTON_ENABLE_HIP)
	list(APPEND PROFILER_SOURCES
		RocTracer/RoctracerProfiler.cpp
		Instrumentation/HipRuntime.cpp
	)
endif()

# Add NPU/Ascend sources if NPU is available
if(PROTON_ENABLE_NPU)
	list(APPEND PROFILER_SOURCES
		Ascend/AscendProfiler.cpp
		Instrumentation/AscendRuntime.cpp
	)
endif()

add_proton_library(ProtonProfiler ${PROFILER_SOURCES})

# Add compile definitions
if(PROTON_ENABLE_CUDA)
	target_compile_definitions(ProtonProfiler PRIVATE PROTON_ENABLE_CUDA)
endif()
if(PROTON_ENABLE_HIP)
	target_compile_definitions(ProtonProfiler PRIVATE PROTON_ENABLE_HIP)
endif()
if(PROTON_ENABLE_NPU)
	target_compile_definitions(ProtonProfiler PRIVATE PROTON_ENABLE_NPU)
endif()
