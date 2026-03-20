#ifndef PROTON_PROFILER_ASCEND_PROFILER_H_
#define PROTON_PROFILER_ASCEND_PROFILER_H_

#include "Profiler/GPUProfiler.h"
#include <string>
#include <cstdint>

namespace proton {

class AscendProfiler : public GPUProfiler<AscendProfiler> {
public:
  AscendProfiler();
  virtual ~AscendProfiler();

  // Internal use: get pImpl for RT callback access
  GPUProfilerPimplInterface* getPimpl() { return pImpl.get(); }

  // Public API for kernel tracking from launcher
  // Returns correlation ID for the kernel launch
  uint64_t recordKernelLaunch(const char* kernelName, uint32_t gridX, 
                               uint32_t gridY, uint32_t gridZ);

  // Record kernel completion with correlation ID
  void recordKernelComplete(uint64_t correlationId);

protected:
  // Profiler interface - override pure virtual methods
  void doSetMode(const std::vector<std::string> &modeAndOptions) override;

private:
  struct AscendProfilerPimpl;

  friend class GPUProfiler<AscendProfiler>;
};

} // namespace proton

#endif // PROTON_PROFILER_ASCEND_PROFILER_H_
