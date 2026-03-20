add_proton_library(ProtonSession
  Session.cpp
)

# Add compile definitions
if(PROTON_ENABLE_CUDA)
	target_compile_definitions(ProtonSession PRIVATE PROTON_ENABLE_CUDA)
endif()
if(PROTON_ENABLE_HIP)
	target_compile_definitions(ProtonSession PRIVATE PROTON_ENABLE_HIP)
endif()
if(PROTON_ENABLE_NPU)
	target_compile_definitions(ProtonSession PRIVATE PROTON_ENABLE_NPU)
endif()
