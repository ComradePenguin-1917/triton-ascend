#ifndef PROTON_PROFILER_ASCEND_PROFILER_H_
#define PROTON_PROFILER_ASCEND_PROFILER_H_

#include "Profiler/GPUProfiler.h"
#include <string>
#include <cstdint>
#include <unordered_map>

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

  // Override to add wall-clock timing for scopes as fallback
  void startOp(const Scope &scope) override;
  void stopOp(const Scope &scope) override;

private:
  struct AscendProfilerPimpl;

  // Per-scope wall-clock start times (fallback when RT callback unavailable)
  std::unordered_map<size_t, uint64_t> scopeStartTimes_;

  friend class GPUProfiler<AscendProfiler>;
};

} // namespace proton

#endif // PROTON_PROFILER_ASCEND_PROFILER_H_
