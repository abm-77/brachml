#include <brachml/Dialect/Basic/BrachMLDialect.h>
#include <brachml/Dialect/Basic/BrachMLOps.h>
#include <brachml/Transforms/Passes.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ThreadPool.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

#include <chrono>
#include <mutex>

namespace brachml {

struct FusionCandidate {
  llvm::SmallVector<mlir::Operation *> ops;
  double score = 0.0;

  bool operator==(const FusionCandidate &o) const { return ops == o.ops; }
};
struct BeamState {
  // Each entry is a group of ops that will be wrapped in a fused_region.
  llvm::SmallVector<llvm::SmallVector<mlir::Operation *>> fusionGroups;
  // Ops already assigned to a group in this plan (can't be fused twice).
  llvm::DenseSet<mlir::Operation *> claimed;
  double totalScore = 0.0;
  bool operator<(const BeamState &o) const { return totalScore < o.totalScore; }
};

static bool isFusableOp(mlir::Operation *op) {
  return op->hasTrait<FusableOpTrait>();
}

static double scoreFusion(llvm::ArrayRef<mlir::Operation *> ops) {
  const auto N = ops.size();
  if (N < 2) return 0.0;

  double score = 0.0;

  // boost score for amount of bytes written to  memory saved
  for (auto i = 0u; i < N - 1; ++i) {
    for (const auto result : ops[i]->getResults()) {
      auto resTy = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
      if (!resTy) continue;

      score += static_cast<double>(resTy.getNumElements()) *
               (resTy.getElementTypeBitWidth() / 8.0);
    }
  }

  // longer fusions are better (up to a point)
  score += log((double)N);

  return score;
}

static llvm::SmallVector<FusionCandidate>
findCandidates(mlir::func::FuncOp func,
               const llvm::DenseSet<mlir::Operation *> &claimed, int maxDepth) {
  llvm::SmallVector<FusionCandidate> candidates;

  // iterate over func body in topo order
  for (auto &op : func.getBody().front()) {
    auto *root = &op;
    if (!isFusableOp(root) || claimed.count(root)) continue;

    llvm::SmallVector<llvm::SmallVector<mlir::Operation *>> chains{{root}};
    for (int i = 1; i < maxDepth; ++i) {
      llvm::SmallVector<llvm::SmallVector<mlir::Operation *>> nextChains;
      for (const auto &chain : chains) {
        llvm::DenseSet<mlir::Operation *> seen;
        for (const auto &result : chain.back()->getResults()) {
          for (mlir::Operation *user : result.getUsers()) {
            if (claimed.count(user) || !isFusableOp(user) ||
                !result.hasOneUse() || !seen.insert(user).second)
              continue;

            llvm::SmallVector<mlir::Operation *> extended = chain;
            extended.push_back(user);
            if (extended.size() >= 2)
              candidates.push_back(
                  FusionCandidate{extended, scoreFusion(extended)});
            nextChains.push_back(std::move(extended));
          }
        }
      }
      chains = std::move(nextChains);
    }
  }
  llvm::sort(candidates.begin(), candidates.end(),
             [](const FusionCandidate &a, const FusionCandidate &b) {
               return a.score > b.score;
             });
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  return candidates;
}

static void applyFusion(mlir::IRRewriter &rewriter,
                        llvm::ArrayRef<mlir::Operation *> ops) {

  llvm::SmallVector<mlir::Value> externalInputs;
  llvm::DenseSet<mlir::Value> seen;
  llvm::DenseSet<mlir::Operation *> opGroup(ops.begin(), ops.end());
  for (const auto op : ops) {
    for (const auto operand : op->getOperands()) {
      if (!opGroup.count(operand.getDefiningOp()) &&
          seen.insert(operand).second) {
        externalInputs.push_back(operand);
      }
    }
  }

  auto resTy = llvm::SmallVector<mlir::Type>(ops.back()->getResultTypes());
  auto loc = ops.front()->getLoc();
  rewriter.setInsertionPoint(ops.front());
  auto fused =
      brachml::FusedRegionOp::create(rewriter, loc, resTy, externalInputs);

  mlir::IRMapping mapping;
  auto *body = new mlir::Block();
  fused.getBody().push_back(body);
  for (const auto &input : externalInputs) {
    auto arg = body->addArgument(input.getType(), input.getLoc());
    mapping.map(input, arg);
  }

  rewriter.setInsertionPointToStart(body);
  for (auto *op : ops) {
    auto *cloned = rewriter.clone(*op, mapping);
    mapping.map(op->getResults(), cloned->getResults());
  }

  auto lastOp = ops.back();
  llvm::SmallVector<mlir::Value> yieldVals;
  for (const auto &result : lastOp->getResults()) {
    yieldVals.push_back(mapping.lookup(result));
  }
  brachml::YieldOp::create(rewriter, lastOp->getLoc(), yieldVals);
  if (fused.verify().failed()) return;

  rewriter.replaceOp(ops.back(), fused.getResults());
  for (int i = (int)ops.size() - 2; i >= 0; --i) rewriter.eraseOp(ops[i]);
}

struct BeamSearchFusionPass
    : public mlir::PassWrapper<BeamSearchFusionPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BeamSearchFusionPass)

  BeamSearchFusionPass() = default;
  BeamSearchFusionPass(const BeamSearchFusionPass &other)
      : mlir::PassWrapper<BeamSearchFusionPass,
                          mlir::OperationPass<mlir::func::FuncOp>>(other) {}

  Option<int> beamWidth{
      *this, "beam-width",
      llvm::cl::desc("Number of candidate plans to keep alive across each "
                     "search step"),
      llvm::cl::init(4)};

  Option<int> maxDepth{
      *this, "max-depth",
      llvm::cl::desc(
          "Maximum op-chain length to consider for a single fusion group"),
      llvm::cl::init(3)};

  Option<int> numThreads{
      *this, "num-threads",
      llvm::cl::desc("Number of threads for Phase 1 beam expansion. "
                     "0 = use hardware_concurrency()"),
      llvm::cl::init(0)};

  Option<int> timeBudgetMs{
      *this, "time-budget-ms",
      llvm::cl::desc("Maximum wall time in milliseconds for Phase 1 beam "
                     "search. 0 = no limit"),
      llvm::cl::init(0)};

  llvm::StringRef getArgument() const override { return "brachml-beam-fusion"; }
  llvm::StringRef getDescription() const override {
    return "BEAM search over op chains to find the highest-scoring fusion "
           "plan";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<BrachMLDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    std::mutex mu;
    llvm::ThreadPoolStrategy strategy =
        numThreads > 0 ? llvm::heavyweight_hardware_concurrency(numThreads)
                       : llvm::hardware_concurrency();
    llvm::DefaultThreadPool pool(strategy);

    using Clock = std::chrono::steady_clock;
    auto deadline = timeBudgetMs > 0
                        ? Clock::now() + std::chrono::milliseconds(timeBudgetMs)
                        : Clock::time_point::max();

    llvm::SmallVector<BeamState> beam = {BeamState{}};
    while (true) {
      if (Clock::now() > deadline) break;
      llvm::SmallVector<BeamState> nextBeam;
      for (const auto &state : beam) {
        // thread per state expansion
        pool.async([&, state] {
          auto candidates = findCandidates(func, state.claimed, maxDepth);
          for (const auto &candidate : candidates) {
            auto successor = state;
            successor.fusionGroups.push_back(candidate.ops);
            for (auto *op : candidate.ops) successor.claimed.insert(op);
            successor.totalScore += candidate.score;

            // lock to modify the nextBeam
            std::lock_guard<std::mutex> lock(mu);
            nextBeam.push_back(std::move(successor));
          }
        });
      }
      pool.wait();

      if (nextBeam.empty()) break;
      llvm::sort(nextBeam.begin(), nextBeam.end(),
                 [&](const BeamState &a, const BeamState &b) {
                   return a.totalScore > b.totalScore;
                 });
      if ((int)nextBeam.size() > beamWidth) nextBeam.resize(beamWidth);
      beam = std::move(nextBeam);
    }

    if (beam.empty()) return;
    const BeamState &winner = beam.front();
    mlir::IRRewriter rewriter(func.getContext());
    for (auto &group : winner.fusionGroups) {
      if (group.size() < 2) continue;
      applyFusion(rewriter, group);
    }
  }
};

std::unique_ptr<mlir::Pass> createBeamSearchFusionPass() {
  return std::make_unique<BeamSearchFusionPass>();
}

} // namespace brachml
