#include "Profiler/Ascend/AclProfParser.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>

namespace proton {

std::string AclProfParser::findLatestProfDir(const std::string &outputPath) {
  std::filesystem::path base(outputPath);
  if (!std::filesystem::exists(base))
    return "";

  if (std::filesystem::exists(base / "device_0"))
    return outputPath;

  std::string latestDir;
  std::string latestName;
  for (const auto &entry : std::filesystem::directory_iterator(base)) {
    if (!entry.is_directory())
      continue;
    std::string name = entry.path().filename().string();
    if (name.find("PROF_") == 0 && name > latestName) {
      latestName = name;
      latestDir = entry.path().string();
    }
  }
  return latestDir;
}

bool AclProfParser::runMsprofImport(const std::string &profDir) {
  const char *cannHome = std::getenv("ASCEND_HOME_PATH");
  if (!cannHome || cannHome[0] == '\0')
    cannHome = std::getenv("ASCEND_TOOLKIT_HOME");
  if (!cannHome || cannHome[0] == '\0')
    return false;

  std::string msprofPy = std::string(cannHome) +
      "/tools/profiler/profiler_tool/analysis/msprof/msprof.py";
  if (!std::filesystem::exists(msprofPy))
    return false;

  std::string cmd = "python3 \"" + msprofPy + "\" import --collection-dir \"" +
                    profDir + "\" >/dev/null 2>&1";
  return std::system(cmd.c_str()) == 0;
}

std::vector<std::vector<std::string>> AclProfParser::querySqlite(
    const std::string &dbPath, const std::string &sql) {
  std::vector<std::vector<std::string>> rows;
  if (!std::filesystem::exists(dbPath))
    return rows;

  std::string cmd = "sqlite3 -separator $'\\t' \"" + dbPath + "\" \"" + sql + "\" 2>/dev/null";
  FILE *pipe = popen(cmd.c_str(), "r");
  if (!pipe)
    return rows;

  char buf[4096];
  while (fgets(buf, sizeof(buf), pipe)) {
    std::string line(buf);
    if (!line.empty() && line.back() == '\n')
      line.pop_back();
    if (line.empty())
      continue;

    std::vector<std::string> fields;
    std::istringstream iss(line);
    std::string field;
    while (std::getline(iss, field, '\t'))
      fields.push_back(field);
    rows.push_back(std::move(fields));
  }
  pclose(pipe);
  return rows;
}

std::vector<AclKernelEntry> AclProfParser::parseSqliteData(const std::string &profDir) {
  std::vector<AclKernelEntry> entries;
  std::filesystem::path profPath(profDir);

  std::map<uint64_t, std::string> kernelNameMap;
  std::string runtimeDb = (profPath / "host" / "sqlite" / "runtime.db").string();
  if (std::filesystem::exists(runtimeDb)) {
    auto nameRows = querySqlite(runtimeDb,
        "SELECT task_id, kernel_name FROM HostTask WHERE kernel_name != 'N/A';");
    for (auto &row : nameRows) {
      if (row.size() < 2)
        continue;
      uint64_t taskId = std::strtoull(row[0].c_str(), nullptr, 10);
      kernelNameMap[taskId] = row[1];
    }
  }

  for (const auto &entry : std::filesystem::directory_iterator(profPath)) {
    if (!entry.is_directory())
      continue;
    std::string dirname = entry.path().filename().string();
    if (dirname.find("device_") != 0)
      continue;

    uint32_t deviceId = 0;
    auto usPos = dirname.find('_');
    if (usPos != std::string::npos)
      deviceId = static_cast<uint32_t>(std::stoul(dirname.substr(usPos + 1)));

    std::string taskDb = (entry.path() / "sqlite" / "ascend_task.db").string();
    std::string metricDb = (entry.path() / "sqlite" / "metric_summary.db").string();

    if (!std::filesystem::exists(taskDb))
      continue;

    auto taskRows = querySqlite(taskDb,
        "SELECT task_id, stream_id, start_time, duration, device_task_type "
        "FROM AscendTask WHERE device_task_type='AI_CORE' ORDER BY start_time;");

    std::map<uint64_t, AclKernelEntry> metricMap;
    if (std::filesystem::exists(metricDb)) {
      auto metricRows = querySqlite(metricDb,
          "SELECT task_id, "
          "aic_mac_ratio_extra, aic_scalar_ratio, aic_mte1_ratio_extra, "
          "aic_mte2_ratio, aic_fixpipe_ratio, aic_icache_miss_rate, "
          "aiv_vec_ratio, aiv_scalar_ratio, aiv_mte1_ratio, "
          "aiv_mte2_ratio, aiv_mte3_ratio, aiv_icache_miss_rate "
          "FROM MetricSummary;");
      for (auto &row : metricRows) {
        if (row.size() < 13)
          continue;
        uint64_t taskId = std::strtoull(row[0].c_str(), nullptr, 10);
        AclKernelEntry m;
        m.aicMacRatio = std::strtod(row[1].c_str(), nullptr);
        m.aicScalarRatio = std::strtod(row[2].c_str(), nullptr);
        m.aicMte1Ratio = std::strtod(row[3].c_str(), nullptr);
        m.aicMte2Ratio = std::strtod(row[4].c_str(), nullptr);
        m.aicFixpipeRatio = std::strtod(row[5].c_str(), nullptr);
        m.aicIcacheMissRate = std::strtod(row[6].c_str(), nullptr);
        m.aivVecRatio = std::strtod(row[7].c_str(), nullptr);
        m.aivScalarRatio = std::strtod(row[8].c_str(), nullptr);
        m.aivMte1Ratio = std::strtod(row[9].c_str(), nullptr);
        m.aivMte2Ratio = std::strtod(row[10].c_str(), nullptr);
        m.aivMte3Ratio = std::strtod(row[11].c_str(), nullptr);
        m.aivIcacheMissRate = std::strtod(row[12].c_str(), nullptr);
        metricMap[taskId] = std::move(m);
      }
    }

    for (auto &row : taskRows) {
      if (row.size() < 5)
        continue;
      AclKernelEntry e;
      e.deviceId = deviceId;
      e.opType = "AI_CORE";
      uint64_t taskId = std::strtoull(row[0].c_str(), nullptr, 10);
      auto nameIt = kernelNameMap.find(taskId);
      e.opName = (nameIt != kernelNameMap.end()) ? nameIt->second : "kernel_task_" + row[0];
      e.streamId = std::strtoull(row[1].c_str(), nullptr, 10);
      e.startTimeNs = static_cast<uint64_t>(std::strtod(row[2].c_str(), nullptr));
      e.durationUs = static_cast<uint64_t>(std::strtod(row[3].c_str(), nullptr));

      auto it = metricMap.find(taskId);
      if (it != metricMap.end()) {
        e.aicMacRatio = it->second.aicMacRatio;
        e.aicScalarRatio = it->second.aicScalarRatio;
        e.aicMte1Ratio = it->second.aicMte1Ratio;
        e.aicMte2Ratio = it->second.aicMte2Ratio;
        e.aicFixpipeRatio = it->second.aicFixpipeRatio;
        e.aicIcacheMissRate = it->second.aicIcacheMissRate;
        e.aivVecRatio = it->second.aivVecRatio;
        e.aivScalarRatio = it->second.aivScalarRatio;
        e.aivMte1Ratio = it->second.aivMte1Ratio;
        e.aivMte2Ratio = it->second.aivMte2Ratio;
        e.aivMte3Ratio = it->second.aivMte3Ratio;
        e.aivIcacheMissRate = it->second.aivIcacheMissRate;
      }
      entries.push_back(std::move(e));
    }
  }

  std::sort(entries.begin(), entries.end(),
            [](const AclKernelEntry &a, const AclKernelEntry &b) {
              return a.startTimeNs < b.startTimeNs;
            });
  return entries;
}

std::vector<AclKernelEntry> AclProfParser::parseProfilingData(const std::string &outputPath) {
  std::string profDir = findLatestProfDir(outputPath);
  if (profDir.empty())
    return {};

  std::filesystem::path taskDb = std::filesystem::path(profDir) / "device_0" / "sqlite" / "ascend_task.db";
  if (!std::filesystem::exists(taskDb))
    runMsprofImport(profDir);

  return parseSqliteData(profDir);
}

} // namespace proton
