project(Proton LANGUAGES CXX)

set(PROTON_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
set(PROTON_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")

# ============ Check for includes =============
# Check if CUDA is available by looking for cuda.h
if(CUPTI_INCLUDE_DIR AND EXISTS "${CUPTI_INCLUDE_DIR}")
  # Check if cuda.h exists in the parent directory (CUDA SDK include dir)
  get_filename_component(CUDA_INCLUDE_DIR "${CUPTI_INCLUDE_DIR}" DIRECTORY)
  if(EXISTS "${CUDA_INCLUDE_DIR}/cuda.h")
    set(PROTON_ENABLE_CUDA ON)
    message(STATUS "CUDA support enabled")
  else()
    message(WARNING "CUPTI directory exists but cuda.h not found, NVIDIA GPU profiling will be disabled")
    set(CUPTI_INCLUDE_DIR "")
    set(PROTON_ENABLE_CUDA OFF)
  endif()
else()
  message(WARNING "CUPTI include directory not defined or does not exist, NVIDIA GPU profiling will be disabled")
  set(CUPTI_INCLUDE_DIR "")
  set(PROTON_ENABLE_CUDA OFF)
endif()

# Check if ROCm/HIP is available
if(ROCTRACER_INCLUDE_DIR AND EXISTS "${ROCTRACER_INCLUDE_DIR}")
  # Check if hip headers exist
  if(EXISTS "${ROCTRACER_INCLUDE_DIR}/../hip/hip_runtime_api.h" OR EXISTS "/opt/rocm/include/hip/hip_runtime_api.h")
    set(PROTON_ENABLE_HIP ON)
    message(STATUS "HIP/ROCm support enabled")
  else()
    message(WARNING "ROCTRACER directory exists but HIP headers not found, AMD GPU profiling will be disabled")
    set(ROCTRACER_INCLUDE_DIR "")
    set(PROTON_ENABLE_HIP OFF)
  endif()
else()
  message(WARNING "ROCTRACER include directory not defined or does not exist, AMD GPU profiling will be disabled")
  set(ROCTRACER_INCLUDE_DIR "")
  set(PROTON_ENABLE_HIP OFF)
endif()

# Check if Ascend/NPU is available
if(ASCEND_INCLUDE_DIR AND EXISTS "${ASCEND_INCLUDE_DIR}")
  # Check if ACL headers exist
  if(EXISTS "${ASCEND_INCLUDE_DIR}/acl/acl.h")
    set(PROTON_ENABLE_NPU ON)
    message(STATUS "Ascend/NPU support enabled")
  else()
    message(WARNING "ASCEND directory exists but ACL headers not found, Ascend NPU profiling will be disabled")
    set(ASCEND_INCLUDE_DIR "")
    set(PROTON_ENABLE_NPU OFF)
  endif()
else()
  # Try default Ascend installation paths
  set(ASCEND_SEARCH_PATHS
    "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include"
    "/usr/local/Ascend/ascend-toolkit/latest/runtime/include"
    "/usr/local/Ascend/ascend-toolkit/8.2.RC1/aarch64-linux/include"
    "/usr/local/Ascend/ascend-toolkit/8.2.RC1/runtime/include"
    "/usr/local/Ascend/acllib/include"
  )

  foreach(SEARCH_PATH ${ASCEND_SEARCH_PATHS})
    if(EXISTS "${SEARCH_PATH}/acl/acl.h")
      set(ASCEND_INCLUDE_DIR "${SEARCH_PATH}")
      set(PROTON_ENABLE_NPU ON)
      message(STATUS "Ascend/NPU support enabled (found at ${SEARCH_PATH})")
      break()
    endif()
  endforeach()

  if(NOT PROTON_ENABLE_NPU)
    message(WARNING "ASCEND include directory not found in any default location, Ascend NPU profiling will be disabled")
    set(ASCEND_INCLUDE_DIR "")
    set(PROTON_ENABLE_NPU OFF)
  endif()
endif()

if(NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "JSON include directory not defined")
endif()

# ============ Dependencies =============
find_package(Python3 REQUIRED Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

# ============ Define a GLOBAL property to store object-libraries ============
set_property(GLOBAL PROPERTY PROTON_LIBS "")

# ============ Define a function to create object libraries ============
function(add_proton_library name)
  add_library(${name} OBJECT ${ARGN})

  target_link_libraries(${name} PRIVATE Python3::Module pybind11::headers)

  # Use system to skip warnings caused by legacy clang compilers
  target_include_directories(${name}
    SYSTEM PRIVATE
      "${ROCTRACER_INCLUDE_DIR}"
      "${ASCEND_INCLUDE_DIR}"
  )

  target_include_directories(${name}
    PRIVATE
      "${CUPTI_INCLUDE_DIR}"
      "${JSON_INCLUDE_DIR}"
      "${PROTON_COMMON_DIR}/include"
      "${PROTON_SRC_DIR}/include"
  )

  # If HIP is AMD-based
  target_compile_definitions(${name} PRIVATE __HIP_PLATFORM_AMD__)

  # Append this library name to the GLOBAL property "PROTON_LIBS"
  set_property(GLOBAL APPEND PROPERTY PROTON_LIBS ${name})
endfunction()

# ============ Add subdirectory with actual code that calls add_proton_library ============
add_subdirectory("${PROTON_COMMON_DIR}")
add_subdirectory("${PROTON_SRC_DIR}")

# ============ Add subdirectory with proton tests ============
if(TRITON_BUILD_UT)
  add_subdirectory(test)
endif()

# ============ Possibly handle macOS specifics ============
if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  # Other platforms build with -flto, but we found that this adds significant overhead to our macos CI without providing a major benefit.
  set(PROTON_PYTHON_LDFLAGS "-undefined dynamic_lookup")
endif()

# ============ Collect all object libraries from property and build final shared lib ============
get_property(_proton_obj_libs GLOBAL PROPERTY PROTON_LIBS)

if(NOT _proton_obj_libs)
  message(WARNING "No object libraries were defined in 'PROTON_LIBS'!")
endif()

set(_proton_obj_sources "")
foreach(_lib IN LISTS _proton_obj_libs)
  list(APPEND _proton_obj_sources $<TARGET_OBJECTS:${_lib}>)
  message(STATUS "Collecting object files from ${_lib}")
endforeach()

add_library(proton SHARED ${_proton_obj_sources})

target_link_libraries(proton PRIVATE Python3::Module)

# Link ACL libraries if NPU support is enabled
if(PROTON_ENABLE_NPU)
  # Derive library path from include path
  get_filename_component(ASCEND_BASE_DIR "${ASCEND_INCLUDE_DIR}" DIRECTORY)

  # Find ACL libraries
  find_library(ACL_LIB ascendcl PATHS "${ASCEND_BASE_DIR}/lib64" "${ASCEND_BASE_DIR}/lib" NO_DEFAULT_PATH)
  find_library(ACLLIB acl_op_compiler PATHS "${ASCEND_BASE_DIR}/lib64" "${ASCEND_BASE_DIR}/lib" NO_DEFAULT_PATH)

  if(ACL_LIB)
    message(STATUS "Found ACL library: ${ACL_LIB}")
    target_link_libraries(proton PRIVATE ${ACL_LIB})
  else()
    message(WARNING "ACL library (ascendcl) not found, linking may fail")
  endif()

  if(ACLLIB)
    message(STATUS "Found ACL op compiler library: ${ACLLIB}")
    target_link_libraries(proton PRIVATE ${ACLLIB})
  endif()
endif()

# Apply any macOS linker flags or extra link options
if(PROTON_PYTHON_LDFLAGS)
  target_link_options(proton PRIVATE ${PROTON_PYTHON_LDFLAGS})
endif()
