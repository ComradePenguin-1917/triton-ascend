#ifndef PROTON_PROFILER_ASCEND_ACL_PROF_PARSER_H_
#define PROTON_PROFILER_ASCEND_ACL_PROF_PARSER_H_

#include <string>
#include <vector>
#include <cstdint>

namespace proton {

struct AclKernelEntry {
  std::string opName;
  std::string opType;
  uint64_t startTimeNs = 0;
  uint64_t durationUs = 0;
  uint32_t deviceId = 0;
  uint64_t streamId = 0;
  // AiC pipeline metrics
  double aicMacRatio = 0.0;
  double aicScalarRatio = 0.0;
  double aicMte1Ratio = 0.0;
  double aicMte2Ratio = 0.0;
  double aicFixpipeRatio = 0.0;
  double aicIcacheMissRate = 0.0;
  // AiV pipeline metrics
  double aivVecRatio = 0.0;
  double aivScalarRatio = 0.0;
  double aivMte1Ratio = 0.0;
  double aivMte2Ratio = 0.0;
  double aivMte3Ratio = 0.0;
  double aivIcacheMissRate = 0.0;
};

/// Parse ACL profiling output directory (post-processed sqlite)
class AclProfParser {
public:
  /// Run msprof post-processing and parse results
  /// @param outputPath The ACL profiling output directory (PROF_* or base dir)
  /// @return Vector of parsed kernel entries, sorted by startTime
  static std::vector<AclKernelEntry> parseProfilingData(const std::string &outputPath);

  /// Find the latest PROF_* directory under outputPath
  static std::string findLatestProfDir(const std::string &outputPath);

  /// Run msprof.py import to post-process raw PROF data into sqlite
  static bool runMsprofImport(const std::string &profDir);

  /// Parse task and metric data from sqlite databases
  static std::vector<AclKernelEntry> parseSqliteData(const std::string &profDir);

  /// Execute a sqlite3 query and return tab-separated rows
  static std::vector<std::vector<std::string>> querySqlite(
      const std::string &dbPath, const std::string &sql);
};

} // namespace proton

#endif // PROTON_PROFILER_ASCEND_ACL_PROF_PARSER_H_
