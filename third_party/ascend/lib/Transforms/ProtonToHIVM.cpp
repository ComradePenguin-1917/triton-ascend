//===----------------------------------------------------------------------===//
// Triton Ascend Proton to HIVM Lowering Pass
//
// Buffer layout (i32 words):
//   [0]     preamble      = 0xdeadbeef
//   [1]     blockId       = 0
//   [2]     procId        = 0
//   [3]     bufSize       (data segment size in bytes)
//   [4-5]   initTime      (i64, GetSysCnt at function entry)
//   [6-7]   preFinalTime  (i64, GetSysCnt before last record)
//   [8-9]   postFinalTime (i64, GetSysCnt at function exit)
//   [10]    countVec[0]   (word count written by unit 0)
//   [11+]   dataSegment   (CycleEntry records, 2 words each)
//
// CycleEntry encoding (2 x i32, 8 bytes):
//   word0 = (isStart?0:0x80000000) | (scopeId<<23) | ((cycle>>32)&0x7FF)
//   word1 = cycle & 0xFFFFFFFF
//===----------------------------------------------------------------------===//

#include "ascend/include/Dialect/TritonAscendProton/Transforms/Passes.h"
#include "ascend/include/Dialect/TritonAscendProton/IR/TritonAscendProtonOps.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#define DEBUG_TYPE "ascend-proton-to-hivm"

namespace {

static constexpr int32_t kHeaderBytes = 40;   // 4*i32 + 3*i64
static constexpr int32_t kTotalUnits = 1;     // AICore = 1 unit
static constexpr int32_t kCountVecBytes = kTotalUnits * 4;
static constexpr int32_t kDataSegmentBytes = 4096;
static constexpr int32_t kScratchMemSize =
    kHeaderBytes + kCountVecBytes + kDataSegmentBytes; // 4140
static constexpr uint32_t kPreamble = 0xdeadbeef;

static constexpr int32_t kOffPreamble = 0;
static constexpr int32_t kOffBlockId = 1;
static constexpr int32_t kOffProcId = 2;
static constexpr int32_t kOffBufSize = 3;
static constexpr int32_t kOffInitTime = 4;
static constexpr int32_t kOffPreFinalTime = 6;
static constexpr int32_t kOffPostFinalTime = 8;
static constexpr int32_t kOffCountVec = 10;
static constexpr int32_t kOffDataSegment = 11;

static constexpr uint32_t kEndBit = 0x80000000u;
static constexpr uint32_t kScopeIdShift = 23;
static constexpr uint32_t kCycleUpperMask = 0x7FFu;

class ProtonRecordConverter {
public:
  ProtonRecordConverter(mlir::Operation *funcOp) : funcOp(funcOp) {}

  mlir::LogicalResult convert()
  {
    mlir::SmallVector<mlir::triton::proton::RecordOp, 16> recordOps;
    funcOp->walk([&](mlir::triton::proton::RecordOp op) {
      recordOps.push_back(op);
    });

    if (recordOps.empty()) {
      return mlir::success();
    }

    assignScopeIds(recordOps);

    if (failed(emitHeader())) {
      return mlir::failure();
    }

    if (failed(emitFooterOps())) {
      return mlir::failure();
    }

    mlir::OpBuilder builder(funcOp->getContext());
    for (auto recordOp : recordOps) {
      if (failed(convertRecordOp(recordOp, builder))) {
        return mlir::failure();
      }
    }

    attachAttrs();

    return mlir::success();
  }

private:
  mlir::Operation *funcOp{nullptr};
  llvm::StringMap<uint32_t> scopeNameToId;
  llvm::SmallVector<std::string, 16> scopeNames;
  uint32_t numScopes{0};
  mlir::Value buffer;
  // Per-block section offset (in i32 words): blockIdx * kScratchMemSize / 4
  // Each block writes to its own section of the proton buffer to avoid
  // concurrent writes from multiple AICore blocks.
  mlir::Value sectionOffset;

  void assignScopeIds(mlir::ArrayRef<mlir::triton::proton::RecordOp> recordOps)
  {
    uint32_t nextId = 0;
    llvm::StringSet<> seen;
    for (auto recordOp : recordOps) {
      mlir::StringRef scope = recordOp.getName();
      if (seen.contains(scope)) {
        continue;
      }
      seen.insert(scope);
      scopeNameToId[scope] = nextId;
      scopeNames.push_back(scope.str());
      nextId++;
    }
    numScopes = nextId;
  }

  mlir::LogicalResult emitHeader()
  {
    mlir::Region &body = funcOp->getRegion(0);
    if (body.empty()) {
      return mlir::success();
    }

    auto ctx = funcOp->getContext();
    auto i32Type = mlir::IntegerType::get(ctx, 32);
    auto i64Type = mlir::IntegerType::get(ctx, 64);
    auto gmSpaceAttr = mlir::hivm::AddressSpaceAttr::get(ctx, mlir::hivm::AddressSpace::GM);
    auto bufType = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, i32Type,
                                         mlir::MemRefLayoutAttrInterface{}, gmSpaceAttr);

    // Insert proton_buf as a function argument BEFORE grid_info args.
    // This pass runs after TritonToLinalg, so BlockPtrAnalysis is no longer active.
    auto funcType = mlir::dyn_cast<mlir::func::FuncOp>(funcOp);
    constexpr unsigned kLaunchGridRank = 3;
    unsigned numArgsBefore = funcType.getNumArguments();
    unsigned argIdx = numArgsBefore > kLaunchGridRank * 2
                          ? numArgsBefore - kLaunchGridRank * 2
                          : numArgsBefore;
    funcType.insertArgument(argIdx, bufType,
                            mlir::DictionaryAttr::get(ctx),
                            funcOp->getLoc());

    auto &entryBlock = body.front();
    buffer = entryBlock.getArgument(argIdx);

    mlir::OpBuilder builder(ctx);
    builder.setInsertionPointToStart(&entryBlock);

    auto loc = funcOp->getLoc();
    auto zeroI64 = builder.create<mlir::arith::ConstantIntOp>(loc, 0, i64Type);
    auto zeroI32 = builder.create<mlir::arith::ConstantIntOp>(loc, 0, i32Type);

    // Use GetBlockIdxOp for per-block section offset.
    // TritonGlobalKernelArgsToHIVMOpPass will later replace program_id args
    // with GetBlockIdxOp and erase the original args, so we cannot reference
    // those args directly. GetBlockIdxOp survives that transformation.
    auto blockIdxI64 = builder.create<mlir::hivm::GetBlockIdxOp>(loc, i64Type);
    auto scratchWordsI64 = builder.create<mlir::arith::ConstantIntOp>(
        loc, static_cast<int64_t>(kScratchMemSize / 4), i64Type);
    auto sectionOffI64 = builder.create<mlir::arith::MulIOp>(
        loc, blockIdxI64, scratchWordsI64);
    auto indexType = builder.getIndexType();
    sectionOffset = builder.create<mlir::arith::IndexCastOp>(
        loc, indexType, sectionOffI64);

    auto preambleVal = builder.create<mlir::arith::ConstantIntOp>(
        loc, static_cast<int64_t>(kPreamble), i32Type);
    storeI32(builder, loc, preambleVal, kOffPreamble);

    auto blockIdVal = builder.create<mlir::arith::TruncIOp>(
        loc, i32Type, blockIdxI64);
    storeI32(builder, loc, blockIdVal, kOffBlockId);

    auto procIdVal = builder.create<mlir::arith::ConstantIntOp>(loc, 0, i32Type);
    storeI32(builder, loc, procIdVal, kOffProcId);

    auto bufSizeVal = builder.create<mlir::arith::ConstantIntOp>(
        loc, static_cast<int64_t>(kDataSegmentBytes), i32Type);
    storeI32(builder, loc, bufSizeVal, kOffBufSize);

    auto initSysCnt = builder.create<mlir::triton::proton::ReadCycleCounterOp>(loc, i64Type);
    storeI64(builder, loc, initSysCnt.getResult(), kOffInitTime);

    storeI64(builder, loc, zeroI64.getResult(), kOffPreFinalTime);
    storeI64(builder, loc, zeroI64.getResult(), kOffPostFinalTime);
    storeI32(builder, loc, zeroI32, kOffCountVec);

    return mlir::success();
  }

  mlir::LogicalResult emitFooterOps()
  {
    if (!buffer) {
      return mlir::success();
    }

    mlir::Region &body = funcOp->getRegion(0);
    if (body.empty()) {
      return mlir::success();
    }

    mlir::SmallVector<mlir::Operation *, 4> returnOps;
    body.walk([&](mlir::Operation *op) {
      mlir::StringRef name = op->getName().getStringRef();
      if (name == "tt.return" || name == "func.return" || name == "return") {
        returnOps.push_back(op);
      }
    });

    for (auto *retOp : returnOps) {
      mlir::OpBuilder builder(retOp);
      auto loc = retOp->getLoc();
      auto ctx = builder.getContext();
      auto i64Type = mlir::IntegerType::get(ctx, 64);

      auto postFinalCnt = builder.create<mlir::triton::proton::ReadCycleCounterOp>(loc, i64Type);
      storeI64(builder, loc, postFinalCnt.getResult(), kOffPostFinalTime);
    }

    return mlir::success();
  }

  void storeI32(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value val, int32_t wordOffset)
  {
    auto offIdx = builder.create<mlir::arith::ConstantIndexOp>(loc, wordOffset);
    auto fullIdx = builder.create<mlir::arith::AddIOp>(loc, sectionOffset, offIdx);
    builder.create<mlir::memref::StoreOp>(loc, val, buffer,
                                          mlir::ValueRange{fullIdx});
  }

  void storeI32Dyn(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value val, mlir::Value wordOffset)
  {
    auto fullIdx = builder.create<mlir::arith::AddIOp>(loc, sectionOffset, wordOffset);
    builder.create<mlir::memref::StoreOp>(loc, val, buffer,
                                          mlir::ValueRange{fullIdx});
  }

  void storeI64(mlir::OpBuilder &builder, mlir::Location loc,
                mlir::Value val, int32_t wordOffset)
  {
    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();

    auto maskLo = builder.create<mlir::arith::ConstantIntOp>(
        loc, 0xFFFFFFFFLL, i64Type);
    auto thirtyTwo = builder.create<mlir::arith::ConstantIntOp>(loc, 32, i64Type);

    auto lo64 = builder.create<mlir::arith::AndIOp>(loc, val, maskLo);
    auto lo32 = builder.create<mlir::arith::TruncIOp>(loc, i32Type, lo64);

    auto hi64shifted = builder.create<mlir::arith::ShRUIOp>(loc, val, thirtyTwo);
    auto hi64 = builder.create<mlir::arith::AndIOp>(loc, hi64shifted, maskLo);
    auto hi32 = builder.create<mlir::arith::TruncIOp>(loc, i32Type, hi64);

    storeI32(builder, loc, lo32, wordOffset);
    storeI32(builder, loc, hi32, wordOffset + 1);
  }

  mlir::LogicalResult convertRecordOp(
      mlir::triton::proton::RecordOp op, mlir::OpBuilder &builder)
  {
    mlir::Location loc = op.getLoc();
    if (mlir::isa<mlir::UnknownLoc>(loc)) {
      loc = builder.getUnknownLoc();
    }

    mlir::MLIRContext *ctx = builder.getContext();
    mlir::StringRef scopeName = op.getName();
    bool isStart = op.getIsStart();

    builder.setInsertionPoint(op);

    auto i64Type = builder.getI64Type();

    auto sysCntOp = builder.create<mlir::triton::proton::ReadCycleCounterOp>(loc, i64Type);
    mlir::Value cycle = sysCntOp.getResult();

    auto it = scopeNameToId.find(scopeName);
    if (it == scopeNameToId.end()) {
      op.emitOpError("missing scopeId for scope: ") << scopeName;
      return mlir::failure();
    }
    uint32_t scopeId = it->second;

    auto scopeIdAttr = mlir::IntegerAttr::get(builder.getI32Type(), scopeId);
    auto isStartAttr = mlir::BoolAttr::get(ctx, isStart);

    builder.create<mlir::hivm::ProtonCircularStoreOp>(
        loc, buffer, sectionOffset, cycle, scopeIdAttr, isStartAttr);

    op.erase();
    return mlir::success();
  }

  void attachAttrs()
  {
    mlir::Operation *parent = funcOp;
    while (parent->getParentOp()) {
      parent = parent->getParentOp();
    }
    auto moduleOp = mlir::dyn_cast<mlir::ModuleOp>(parent);
    if (!moduleOp) {
      return;
    }
    auto ctx = moduleOp->getContext();
    auto i32Type = mlir::IntegerType::get(ctx, 32);

    mlir::SmallVector<mlir::Attribute, 16> nameAttrs;
    for (const auto &name : scopeNames) {
      nameAttrs.push_back(mlir::StringAttr::get(ctx, name));
    }
    funcOp->setAttr("proton_scope_names", mlir::ArrayAttr::get(ctx, nameAttrs));

    funcOp->setAttr(
        "proton_num_scopes",
        mlir::IntegerAttr::get(i32Type, numScopes));
    funcOp->setAttr(
        "proton_scratch_size",
        mlir::IntegerAttr::get(i32Type, kScratchMemSize));
    funcOp->setAttr(
        "proton_total_units",
        mlir::IntegerAttr::get(i32Type, kTotalUnits));
  }
};

struct MetricOpToHIVMLowering
    : public mlir::OpRewritePattern<mlir::triton::proton::MetricOp> {
  using mlir::OpRewritePattern<mlir::triton::proton::MetricOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::triton::proton::MetricOp op,
      mlir::PatternRewriter &rewriter) const override
  {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct BarrierOpToHIVMLowering
    : public mlir::OpRewritePattern<mlir::triton::proton::BarrierOp> {
  using mlir::OpRewritePattern<mlir::triton::proton::BarrierOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::triton::proton::BarrierOp op,
      mlir::PatternRewriter &rewriter) const override
  {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ReadCycleCounterToGetSysCntLowering
    : public mlir::OpRewritePattern<mlir::triton::proton::ReadCycleCounterOp> {
  using mlir::OpRewritePattern<mlir::triton::proton::ReadCycleCounterOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::triton::proton::ReadCycleCounterOp op,
      mlir::PatternRewriter &rewriter) const override
  {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto i64Type = mlir::IntegerType::get(ctx, 64);

    // Insert PipeBarrierOp(PIPE_ALL) before GetSysCntOp to ensure all prior
    // pipeline operations complete before reading the cycle counter.
    // Without this barrier, the AICore VLIW scheduler may bundle GetSysCnt
    // with compute/DMA instructions in the same issue slot, causing the
    // measured cycle count to reflect a point before the measured operation
    // actually completes. This is the same pattern used by bishengir's Debug
    // library (pipe_barrier(PIPE_ALL) before every timed operation).
    auto pipeAllAttr = mlir::hivm::PipeAttr::get(ctx, mlir::hivm::PIPE::PIPE_ALL);
    rewriter.create<mlir::hivm::PipeBarrierOp>(loc, pipeAllAttr);

    auto getSysCnt = rewriter.create<mlir::hivm::GetSysCntOp>(loc, i64Type);
    rewriter.replaceOp(op, getSysCnt.getResult());
    return mlir::success();
  }
};

struct TritonAscendProtonToHIVMPass
    : public mlir::PassWrapper<TritonAscendProtonToHIVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonAscendProtonToHIVMPass)

  mlir::StringRef getArgument() const final
  {
    return "ascend-proton-to-hivm";
  }

  mlir::StringRef getDescription() const final
  {
    return "Lower Triton Ascend Proton ops to HIVM profiling annotations";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override
  {
    registry.insert<mlir::hivm::HIVMDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override
  {
    mlir::ModuleOp module = getOperation();

    bool failed = false;
    module.walk([&](mlir::Operation *funcOp) {
      mlir::StringRef opName = funcOp->getName().getStringRef();
      if (opName != "tt.func" && opName != "func.func") {
        return mlir::WalkResult::advance();
      }

      ProtonRecordConverter converter(funcOp);
      if (mlir::failed(converter.convert())) {
        failed = true;
        return mlir::WalkResult::interrupt();
      }

      mlir::MLIRContext *ctx = &getContext();
      mlir::RewritePatternSet patterns(ctx);
      mlir::triton::proton::populateTritonAscendProtonToHIVMPatterns(patterns);

      if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
              funcOp, std::move(patterns)))) {
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

struct TritonAscendProtonLowerCycleCounterPass
    : public mlir::PassWrapper<TritonAscendProtonLowerCycleCounterPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonAscendProtonLowerCycleCounterPass)

  mlir::StringRef getArgument() const final
  {
    return "ascend-proton-lower-cycle-counter";
  }

  mlir::StringRef getDescription() const final
  {
    return "Lower ReadCycleCounterOp to GetSysCntOp (must run after CSE)";
  }

  void runOnOperation() override
  {
    mlir::ModuleOp module = getOperation();
    module.walk([&](mlir::Operation *funcOp) {
      mlir::StringRef opName = funcOp->getName().getStringRef();
      if (opName != "tt.func" && opName != "func.func") {
        return mlir::WalkResult::advance();
      }

      mlir::MLIRContext *ctx = funcOp->getContext();
      mlir::RewritePatternSet patterns(ctx);
      patterns.add<ReadCycleCounterToGetSysCntLowering>(ctx);

      if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
              funcOp, std::move(patterns)))) {
        this->signalPassFailure();
        return mlir::WalkResult::interrupt();
      }

      return mlir::WalkResult::advance();
    });
  }
};

} // namespace

void mlir::triton::proton::populateTritonAscendProtonToHIVMPatterns(
    mlir::RewritePatternSet &patterns)
{
  patterns.add<MetricOpToHIVMLowering, BarrierOpToHIVMLowering>(
      patterns.getContext());
}

std::unique_ptr<mlir::Pass>
mlir::triton::proton::createTritonAscendProtonToHIVMPass()
{
  return std::make_unique<TritonAscendProtonToHIVMPass>();
}

std::unique_ptr<mlir::Pass>
mlir::triton::proton::createTritonAscendProtonLowerCycleCounterPass()
{
  return std::make_unique<TritonAscendProtonLowerCycleCounterPass>();
}

void mlir::triton::proton::registerTritonAscendProtonPasses()
{
  mlir::triton::proton::registerProtonToHIVMPass();
}
