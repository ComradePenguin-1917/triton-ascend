//===----------------------------------------------------------------------===//
// Triton Ascend Proton to HIVM Lowering Pass
//===----------------------------------------------------------------------===//

#include "ascend/include/Dialect/TritonAscendProton/Transforms/Passes.h"
#include "ascend/include/Dialect/TritonAscendProton/IR/TritonAscendProtonOps.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"

#define DEBUG_TYPE "ascend-proton-to-hivm"

// using namespace mlir;
// using namespace mlir::triton::proton;
// using namespace mlir::hivm;

namespace
{

//===----------------------------------------------------------------------===//
// Helper to convert Proton records to HIVM profiling ops
//===----------------------------------------------------------------------===//

// This Pass handles RecordOp by:
// 1. For "start" records: Insert PipeBarrierOp + GetSysCntOp, store the timestamp
// 2. For "end" records: Insert PipeBarrierOp + GetSysCntOp, compute delta, print via DebugOp

class ProtonRecordConverter {
public:
  ProtonRecordConverter(mlir::Operation* funcOp)
    :funcOp(funcOp)
  {
  }

  mlir::LogicalResult convert()
  {
    mlir::SmallVector<mlir::triton::proton::RecordOp, 16> recordOps;
    this->funcOp->walk([&](mlir::triton::proton::RecordOp op) {
      recordOps.push_back(op);
    });

    if (recordOps.empty()) {
      return mlir::success();
    }

    this->initWorkspaceLayout(recordOps);
    if (mlir::failed(this->createTimingBuffer())) {
      return mlir::failure();
    }

    // Process each RecordOp
    mlir::OpBuilder builder(this->funcOp->getContext());
    for (auto recordOp : recordOps) {
      if (failed(convertRecordOp(recordOp, builder))) {
        return mlir::failure();
      }
    }

    return mlir::success();
  }

private:
  mlir::Operation *funcOp{nullptr};
  // Map from scope name to the Value holding the start timestamp
  mlir::DenseMap<mlir::StringRef, mlir::Value> scopeStartTimestamps;
  // Map a timing scope to its slot offset inside the GM buffer
  mlir::DenseMap<mlir::StringRef, uint32_t> scopeSlotIndex;
  // AllocWorkspace-backed memref that persists timestamps to GM
  mlir::Value timingBuffer;
  uint32_t allocatedSlots{0};

  bool useMockTimestamps{true};
  int64_t nextMockTimestamp{1000};
  int64_t mockTimestampStep{128};


  void initWorkspaceLayout(mlir::ArrayRef<mlir::triton::proton::RecordOp> recordOps)
  {
    uint32_t nextSlot = 0;

    for (mlir::triton::proton::RecordOp recordOp : recordOps) {
      mlir::StringRef scope = recordOp.getName();
      if (scopeSlotIndex.contains(scope)) {
        continue;
      }
      this->scopeSlotIndex[scope] = nextSlot;
      nextSlot += 2;  // Reserve slots for start + delta/end per scope
    }

    this->allocatedSlots = nextSlot;
  }

  mlir::LogicalResult createTimingBuffer()
  {
    if (this->allocatedSlots == 0 || this->funcOp->getNumRegions() == 0) {
      return mlir::success();
    }

    mlir::Region& body = this->funcOp->getRegion(0);
    if (body.empty()) {
      return mlir::success();
    }

    mlir::OpBuilder builder(this->funcOp->getContext());
    builder.setInsertionPointToStart(&body.front());

    auto loc = this->funcOp->getLoc();
    auto i64Type = builder.getI64Type();
    auto memrefType =
      mlir::MemRefType::get({static_cast<int64_t>(allocatedSlots)}, i64Type);
    auto alloc =
      builder.create<bishengir::memref_ext::AllocWorkspaceOp>(
          loc, memrefType, mlir::Value(), mlir::ValueRange{}, mlir::ValueRange{});

    this->timingBuffer = alloc.getResult();

    llvm::errs() << "[ProtonToHIVM] Created buffer with " << allocatedSlots << " slots\n";

    return mlir::success();
  }

  void storeToBuffer(
      mlir::OpBuilder& builder, mlir::Location loc, mlir::Value data, uint32_t offset)
  {
    if (!this->timingBuffer) {
      return;
    }

    auto index = builder.create<mlir::arith::ConstantIndexOp>(loc, offset);
    builder.create<mlir::memref::StoreOp>(
        loc, data, this->timingBuffer, mlir::ValueRange{index});
  }

  mlir::Value createMockTimestamp(mlir::OpBuilder& builder, mlir::Location loc)
  {
     auto i64Type = builder.getI64Type();
     auto constant =
       builder.create<mlir::arith::ConstantIntOp>(
           loc, this->nextMockTimestamp, i64Type);

     this->nextMockTimestamp += this->mockTimestampStep;

     return constant;
  }

  mlir::LogicalResult convertRecordOp(
      mlir::triton::proton::RecordOp op, mlir::OpBuilder &builder)
  {
    mlir::Location loc = op.getLoc();
    if (mlir::isa<mlir::UnknownLoc>(loc)) {
      loc = builder.getUnknownLoc();
    }

    mlir::MLIRContext* ctx = builder.getContext();
    mlir::StringRef scopeName = op.getName();
    bool isStart = op.getIsStart();

    // Debug output
    llvm::errs() << "[ProtonToHIVM] Converting " << (isStart ? "start" : "end")
                 << " record: " << scopeName << "\n";

    builder.setInsertionPoint(op);

    // Create PipeBarrierOp with PIPE_ALL to serialize execution
    auto pipeAttr = mlir::hivm::PipeAttr::get(ctx, mlir::hivm::PIPE::PIPE_ALL);
    builder.create<mlir::hivm::PipeBarrierOp>(loc, pipeAttr);
    llvm::errs() << "[ProtonToHIVM] Created PipeBarrierOp\n";

    // Create GetSysCntOp to get current timestamp
    auto i64Type = builder.getI64Type();
    auto sysCntOp = builder.create<mlir::hivm::GetSysCntOp>(loc, i64Type);
    // mlir::Value timestamp = sysCntOp.getResult();
    // llvm::errs() << "[ProtonToHIVM] Created GetSysCntOp\n";

    mlir::Value timestamp = this->createMockTimestamp(builder, loc);
    llvm::errs() << "[ProtonToHIVM] Created mock timestamp\n";

    auto slotIt = scopeSlotIndex.find(scopeName);
    if (slotIt == scopeSlotIndex.end()) {
      op.emitOpError("missing workspace slot for scope");
      return llvm::failure();
    }
    uint32_t baseSlot = slotIt->second;

    if (isStart) {
      // Store the start timestamp for this scope
      scopeStartTimestamps[scopeName] = timestamp;
      this->storeToBuffer(builder, loc, timestamp, baseSlot);
    } else {
      // This is an end record - compute the time delta
      auto it = scopeStartTimestamps.find(scopeName);
      if (it != scopeStartTimestamps.end()) {
        mlir::Value startTimestamp = it->second;

        // Compute delta: end_cnt - start_cnt
        auto delta = builder.create<mlir::arith::SubIOp>(loc, timestamp, startTimestamp);
        llvm::errs() << "[ProtonToHIVM] Computed delta for " << scopeName << "\n";

        this->storeToBuffer(builder, loc, timestamp, baseSlot);
        this->storeToBuffer(builder, loc, delta.getResult(), baseSlot + 1);

        llvm::errs() << "[ProtonToHIVM] Stored timing result for " << scopeName
                     << "\n";

        // Clean up the start timestamp from the map
        scopeStartTimestamps.erase(it);
      }
    }

    // Erase the original RecordOp
    op.erase();
    return llvm::success();
  }
};

//===----------------------------------------------------------------------===//
// Additional lowering patterns for other Proton ops
//===----------------------------------------------------------------------===//

// Pattern to lower MetricOp - just erase for now
struct MetricOpToHIVMLowering
  :public mlir::OpRewritePattern<mlir::triton::proton::MetricOp>
{
  using mlir::OpRewritePattern<mlir::triton::proton::MetricOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::triton::proton::MetricOp op, mlir::PatternRewriter& rewriter) const override
  {
    // HIVM metrics are handled at runtime, not at compile time
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// Pattern to lower BarrierOp - just erase for now
struct BarrierOpToHIVMLowering
  :public mlir::OpRewritePattern<mlir::triton::proton::BarrierOp>
{
  using mlir::OpRewritePattern<mlir::triton::proton::BarrierOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::triton::proton::BarrierOp op, mlir::PatternRewriter &rewriter) const override
  {
    // TODO: Map to HIVM PipeBarrierOp
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct TritonAscendProtonToHIVMPass
  :public mlir::PassWrapper<TritonAscendProtonToHIVMPass, mlir::OperationPass<mlir::ModuleOp>>
{
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonAscendProtonToHIVMPass)

  mlir::StringRef getArgument() const final
  {
    return "ascend-proton-to-hivm";
  }

  mlir::StringRef getDescription() const final
  {
    return "Lower Triton Ascend Proton ops to HIVM profiling annotations";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const override
  {
    registry.insert<mlir::hivm::HIVMDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override
  {
    mlir::ModuleOp module = getOperation();

    // Process all functions in the module (including tt.func and func.func)
    bool failed = false;
    module.walk([&](mlir::Operation* funcOp) {
        mlir::StringRef opName = funcOp->getName().getStringRef();
      // Handle both func.func and tt.func
      if (opName != "tt.func" && opName != "func.func") {
        return mlir::WalkResult::advance();
      }

      // Convert RecordOps to HIVM profiling operations
      ProtonRecordConverter converter(funcOp);
      if (mlir::failed(converter.convert())) {
        failed = true;
        return mlir::WalkResult::interrupt();
      }

      // Apply other lowering patterns for remaining Proton ops
      mlir::MLIRContext* ctx = &getContext();
      mlir::RewritePatternSet patterns(ctx);
      mlir::triton::proton::populateTritonAscendProtonToHIVMPatterns(patterns);

      if (mlir::failed(mlir::applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        failed = true;
        return mlir::WalkResult::interrupt();
      }

      return mlir::WalkResult::advance();
    });

    if (failed) {
      this->signalPassFailure();
    }
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::triton::proton::populateTritonAscendProtonToHIVMPatterns(
    mlir::RewritePatternSet& patterns)
{
  // Note: RecordOp is handled by ProtonRecordConverter, not pattern rewriting
  patterns.add<MetricOpToHIVMLowering, BarrierOpToHIVMLowering>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> mlir::triton::proton::createTritonAscendProtonToHIVMPass()
{
  return std::make_unique<TritonAscendProtonToHIVMPass>();
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

void mlir::triton::proton::registerTritonAscendProtonPasses()
{
  mlir::triton::proton::registerProtonToHIVMPass();
}
