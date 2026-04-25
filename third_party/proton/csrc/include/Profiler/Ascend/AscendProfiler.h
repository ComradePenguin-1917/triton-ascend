#ifndef PROTON_PROFILER_ASCEND_PROFILER_H_
#define PROTON_PROFILER_ASCEND_PROFILER_H_

#include "Profiler/GPUProfiler.h"
#include <string>
#include <cstdint>
#include <unordered_map>

namespace proton {

class AscendProfiler : public GPUProfiler<AscendProfiler>, public ScopeInterface {
public:
  AscendProfiler();
  virtual ~AscendProfiler();

protected:
  // Profiler interface - override pure virtual methods
  void doSetMode(const std::vector<std::string> &modeAndOptions) override;

  // Override to add wall-clock timing for scopes as fallback
  void startOp(const Scope &scope) override;
  void stopOp(const Scope &scope) override;

  // ScopeInterface - wall-clock timing for proton.scope()
  void enterScope(const Scope &scope) override;
  void exitScope(const Scope &scope) override;

private:
  struct AscendProfilerPimpl;

  // Per-scope wall-clock start times
  std::unordered_map<size_t, uint64_t> scopeStartTimes_;

  friend class GPUProfiler<AscendProfiler>;
};

} // namespace proton

#endif // PROTON_PROFILER_ASCEND_PROFILER_H_
