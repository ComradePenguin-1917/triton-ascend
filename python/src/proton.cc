#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "ascend/include/Dialect/TritonAscendProton/IR/TritonAscendProtonDialect.h"
#include "ascend/include/Dialect/TritonAscendProton/IR/TritonAscendProtonOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/StringRef.h"

namespace py = pybind11;

// namespace mlir {
// namespace triton {
// namespace proton {
// enum class MetricType {
//   CYCLE = 0,
//   AICore = 1,
//   AICPU = 2,
//   Memory = 3,
//   Sync = 4,
// };
//
// enum class SamplingStrategy {
//   NONE = 0,
//   SELECTIVE = 1,
//   FULL = 2,
// };
//
// enum class Granularity {
//   CTA = 0,
//   WARP = 1,
//   WARP_2 = 2,
//   WARP_4 = 3,
//   WARP_8 = 4,
//   WARP_GROUP = 5,
//   WARP_GROUP_2 = 6,
//   WARP_GROUP_4 = 7,
//   WARP_GROUP_8 = 8,
//   AICORE = 9,
//   BLOCK = 10,
//   KERNEL = 11,
// };
//
// enum class BufferStrategy {
//   CIRCULAR = 0,
//   FLUSH = 1,
// };
//
// enum class BufferType {
//   SHARED = 0,
//   GLOBAL = 1,
// };
// } // namespace mlir::triton::proton
// } // namespace mlir::triton
// } // namespace mlir

void init_triton_proton(py::module &&m) {
  using namespace mlir::triton::proton;

  m.doc() = "Python bindings to the Ascend Proton backend";

  py::enum_<MetricType>(m, "METRIC_TYPE", py::module_local())
      .value("CYCLE", MetricType::Cycle)
      .value("AICORE", MetricType::AICore)
      .value("AICPU", MetricType::AICPU)
      .value("MEMORY", MetricType::Memory)
      .value("SYNC", MetricType::Sync)
      .export_values();

  py::enum_<SamplingStrategy>(m, "SAMPLING_STRATEGY", py::module_local())
      .value("NONE", SamplingStrategy::None)
      .value("SELECTIVE", SamplingStrategy::Selective)
      .value("FULL", SamplingStrategy::Full)
      .export_values();

  py::enum_<Granularity>(m, "GRANULARITY", py::module_local())
      // .value("CTA", Granularity::CTA)
      // .value("WARP", Granularity::WARP)
      // .value("WARP_2", Granularity::WARP_2)
      // .value("WARP_4", Granularity::WARP_4)
      // .value("WARP_8", Granularity::WARP_8)
      // .value("WARP_GROUP", Granularity::WARP_GROUP)
      // .value("WARP_GROUP_2", Granularity::WARP_GROUP_2)
      // .value("WARP_GROUP_4", Granularity::WARP_GROUP_4)
      // .value("WARP_GROUP_8", Granularity::WARP_GROUP_8)
      .value("AICORE", Granularity::AICore)
      .value("BLOCK", Granularity::Block)
      .value("KERNEL", Granularity::Kernel)
      .export_values();

  py::enum_<BufferStrategy>(m, "BUFFER_STRATEGY", py::module_local())
      .value("CIRCULAR", BufferStrategy::Circular)
      .value("FLUSH", BufferStrategy::Flush)
      .export_values();

  py::enum_<BufferType>(m, "BUFFER_TYPE", py::module_local())
      .value("GLOBAL_MEMORY", BufferType::GlobalMemory)
      .value("LOCAL_MEMORY", BufferType::LocalMemory)
      .value("UNIFIED_BUFFER", BufferType::UnifiedBuffer)
      .value("L1_BUFFER", BufferType::L1Buffer)
      .export_values();

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    mlir::triton::proton::registerTritonAscendProtonDialect(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("get_scope_id_names", [](py::object) {
    return std::vector<std::pair<size_t, std::string>>{};
  });

  m.def("get_scope_id_parents", [](py::object) {
    return std::vector<std::pair<size_t, size_t>>{};
  });

  m.def("create_proton_record",
        [](py::capsule builderCapsule, bool isStart,
           const std::string &name) -> void {
          auto *opBuilder =
              static_cast<mlir::OpBuilder *>(builderCapsule.get_pointer());
          if (!opBuilder) {
            throw std::runtime_error(
                "Invalid builder capsule: expected mlir::OpBuilder");
          }
          opBuilder->create<mlir::triton::proton::RecordOp>(
              opBuilder->getUnknownLoc(), isStart, llvm::StringRef(name),
              nullptr, nullptr);
        });

  m.def("add_convert_proton_to_protongpu",
        [](py::object, MetricType, SamplingStrategy, const std::string &,
           Granularity, BufferStrategy, BufferType, int32_t, int32_t, int64_t,
           int32_t, bool) {});

  m.def("add_allocate_proton_shared_memory", [](py::object) {});

  m.def("add_allocate_proton_global_scratch_buffer", [](py::object) {});

  m.def("add_schedule_buffer_store", [](py::object) {});

  m.def("add_sched_barriers", [](py::object) {});
}
