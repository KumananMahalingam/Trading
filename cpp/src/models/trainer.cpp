#include "trading/util/logging.hpp"
// === FILE: cpp/src/models/trainer.cpp ===
#include "trading/models/trainer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>


#include "trading/config/settings.hpp"
#include "trading/models/losses.hpp"

namespace trading::models {

namespace {

// Snapshot/restore model parameters + buffers for "best model" early stopping.
using StateDict = std::unordered_map<std::string, torch::Tensor>;

StateDict snapshot(const ReturnPredictor& model) {
    StateDict state;
    for (const auto& p : model->named_parameters()) {
        state[p.key()] = p.value().detach().clone();
    }
    for (const auto& b : model->named_buffers()) {
        state[b.key()] = b.value().detach().clone();
    }
    return state;
}

void restore(ReturnPredictor& model, const StateDict& state) {
    torch::NoGradGuard no_grad;
    for (auto& p : model->named_parameters()) {
        auto it = state.find(p.key());
        if (it != state.end()) {
            p.value().copy_(it->second);
        }
    }
    for (auto& b : model->named_buffers()) {
        auto it = state.find(b.key());
        if (it != state.end()) {
            b.value().copy_(it->second);
        }
    }
}

int adaptive_batch_size(int64_t num_train_samples) {
    const int candidate = static_cast<int>(num_train_samples / 10);
    return std::min(config::kBatchSizeMax,
                    std::max(config::kBatchSizeMin, candidate));
}

void set_lr(torch::optim::AdamW& optimizer, double lr) {
    for (auto& group : optimizer.param_groups()) {
        static_cast<torch::optim::AdamWOptions&>(group.options()).lr(lr);
    }
}

// Cosine-annealing schedule from base_lr down to eta_min over t_max epochs.
double cosine_lr(double base_lr, double eta_min, int epoch, int t_max) {
    constexpr double kPi = 3.14159265358979323846;
    const double cos_term =
        1.0 + std::cos(kPi * static_cast<double>(epoch) / static_cast<double>(t_max));
    return eta_min + 0.5 * (base_lr - eta_min) * cos_term;
}

}  // namespace

Result<TrainingResult> train_and_evaluate(ReturnPredictor& model,
                                          const PreparedData& data,
                                          const torch::Device& device) {
    if (data.train.empty()) {
        return Result<TrainingResult>::err(
            Error(Error::Code::InsufficientData, "Empty training split"));
    }

    model->to(device);

    HybridLoss criterion(config::kLossAlpha, config::kLossBeta, config::kLossGamma,
                         config::kLargeMoveThreshold);

    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(config::kLearningRate)
            .weight_decay(config::kWeightDecay)
            .betas(std::make_tuple(0.9, 0.999)));

    const int64_t num_train = data.train.size();
    const int batch_size = adaptive_batch_size(num_train);

    double best_val_loss = std::numeric_limits<double>::infinity();
    int best_epoch = 0;
    int patience_counter = 0;
    StateDict best_state = snapshot(model);
    const bool has_val = !data.val.empty();

    log::info(log::cat("Training: ", num_train, " samples, batch_size ",
                          batch_size, ", ", config::kNumEpochs, " epochs"));

    for (int epoch = 0; epoch < config::kNumEpochs; ++epoch) {
        set_lr(optimizer, cosine_lr(config::kLearningRate, config::kCosineEtaMin,
                                    epoch, config::kNumEpochs));

        // ---- Train ----
        model->train();
        auto perm = torch::randperm(num_train,
                                    torch::TensorOptions().dtype(torch::kLong).device(device));
        double train_loss = 0.0;
        int64_t num_batches = 0;

        for (int64_t start = 0; start < num_train; start += batch_size) {
            const int64_t end = std::min(start + batch_size, num_train);
            // Skip degenerate single-sample batches (BatchNorm/var stability).
            if (end - start < 2) {
                continue;
            }
            auto idx = perm.slice(0, start, end);
            auto xb = data.train.x.index_select(0, idx);
            auto yb = data.train.y.index_select(0, idx);

            optimizer.zero_grad();
            auto pred = model->forward(xb);
            auto loss = criterion(pred, yb);
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(),
                                              config::kGradientClipNorm);
            optimizer.step();

            train_loss += loss.item<double>();
            ++num_batches;
        }
        train_loss /= std::max<int64_t>(1, num_batches);

        // ---- Validate ----
        double val_loss = train_loss;
        if (has_val) {
            model->eval();
            torch::NoGradGuard no_grad;
            auto pred = model->forward(data.val.x);
            val_loss = criterion(pred, data.val.y).item<double>();
        }

        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            log::info(log::cat("Epoch [", epoch + 1, "/", config::kNumEpochs,
                                  "]  train_loss=", log::fixed(train_loss, 6),
                                  "  val_loss=", log::fixed(val_loss, 6)));
        }

        // ---- Early stopping on val loss ----
        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            best_epoch = epoch;
            patience_counter = 0;
            best_state = snapshot(model);
        } else {
            ++patience_counter;
            if (patience_counter >= config::kPatience) {
                log::info(log::cat("Early stopping at epoch ", epoch + 1,
                                      " (best epoch ", best_epoch + 1, ")"));
                break;
            }
        }
    }

    // Restore best weights.
    restore(model, best_state);
    log::info(log::cat("Training complete. Best epoch ", best_epoch + 1,
                          ", val_loss ", log::fixed(best_val_loss, 6)));

    // ---- Evaluate on test set with Monte-Carlo dropout ----
    TrainingResult result;
    result.best_epoch = best_epoch;
    result.best_val_loss = best_val_loss;

    if (data.test.empty()) {
        log::warn("Test split is empty; returning zeroed metrics");
        return Result<TrainingResult>::ok(std::move(result));
    }

    model->eval();
    auto [mean_pred, std_pred] = model->predict_mc(data.test.x, config::kMcSamples);
    mean_pred = mean_pred.to(torch::kCPU).contiguous();
    std_pred = std_pred.to(torch::kCPU).contiguous();
    auto actual = data.test.y.to(torch::kCPU).contiguous();

    const int64_t n = mean_pred.size(0);
    auto pred_acc = mean_pred.accessor<float, 2>();
    auto std_acc = std_pred.accessor<float, 2>();
    auto act_acc = actual.accessor<float, 2>();

    result.predictions.reserve(static_cast<std::size_t>(n));
    result.actuals.reserve(static_cast<std::size_t>(n));
    result.uncertainties.reserve(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        result.predictions.push_back(static_cast<double>(pred_acc[i][0]));
        result.actuals.push_back(static_cast<double>(act_acc[i][0]));
        result.uncertainties.push_back(static_cast<double>(std_acc[i][0]));
    }
    result.dates = data.test_dates;

    result.metrics =
        compute_metrics(result.predictions, result.actuals, result.uncertainties);

    return Result<TrainingResult>::ok(std::move(result));
}

}  // namespace trading::models
