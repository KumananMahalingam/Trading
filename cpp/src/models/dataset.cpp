#include "trading/util/logging.hpp"
// === FILE: cpp/src/models/dataset.cpp ===
#include "trading/models/dataset.hpp"

#include <vector>


#include "trading/config/settings.hpp"

namespace trading::models {

namespace {

// Build a windowed tensor split from a scaled feature matrix and raw targets,
// using only the contiguous row range [row_begin, row_end).
//
//   sample i (local) -> x = scaled[row_begin+i : row_begin+i+window]
//                       y = target[row_begin+i+window]
TensorSplit build_split(const Eigen::MatrixXd& scaled,
                        const std::vector<double>& target,
                        std::size_t row_begin,
                        std::size_t row_end,
                        int window,
                        const torch::Device& device) {
    const int num_features = static_cast<int>(scaled.cols());
    const std::size_t span = (row_end > row_begin) ? (row_end - row_begin) : 0;

    TensorSplit split;
    if (span <= static_cast<std::size_t>(window)) {
        split.x = torch::empty({0, window, num_features}, torch::kFloat32);
        split.y = torch::empty({0, 1}, torch::kFloat32);
        return split;
    }

    const std::size_t num_samples = span - static_cast<std::size_t>(window);

    auto x = torch::empty(
        {static_cast<int64_t>(num_samples), window, num_features}, torch::kFloat32);
    auto y = torch::empty({static_cast<int64_t>(num_samples), 1}, torch::kFloat32);

    auto x_acc = x.accessor<float, 3>();
    auto y_acc = y.accessor<float, 2>();

    for (std::size_t i = 0; i < num_samples; ++i) {
        const std::size_t start = row_begin + i;
        for (int w = 0; w < window; ++w) {
            for (int f = 0; f < num_features; ++f) {
                x_acc[static_cast<int64_t>(i)][w][f] =
                    static_cast<float>(scaled(static_cast<Eigen::Index>(start + w), f));
            }
        }
        const std::size_t target_idx = start + static_cast<std::size_t>(window);
        y_acc[static_cast<int64_t>(i)][0] = static_cast<float>(target[target_idx]);
    }

    split.x = x.to(device);
    split.y = y.to(device);
    return split;
}

}  // namespace

Result<PreparedData> prepare_data(const features::FeatureMatrix& fm,
                                  int window_size,
                                  const torch::Device& device) {
    const std::size_t n = fm.size();
    if (n < static_cast<std::size_t>(window_size) + 10) {
        return Result<PreparedData>::err(
            Error(Error::Code::InsufficientData,
                  "Not enough rows to prepare windows"));
    }

    // Temporal 60/20/20 split on rows.
    const std::size_t n_train =
        static_cast<std::size_t>(static_cast<double>(n) * config::kTrainFraction);
    const std::size_t n_val =
        static_cast<std::size_t>(static_cast<double>(n) * config::kValFraction);
    const std::size_t train_end = n_train;
    const std::size_t val_end = n_train + n_val;
    const std::size_t test_end = n;

    // Pack features into an Eigen matrix [n x num_features].
    const int num_features = fm.num_features();
    Eigen::MatrixXd feature_mat(static_cast<Eigen::Index>(n), num_features);
    for (std::size_t r = 0; r < n; ++r) {
        for (int f = 0; f < num_features; ++f) {
            feature_mat(static_cast<Eigen::Index>(r), f) =
                fm.features[r][static_cast<std::size_t>(f)];
        }
    }

    // Fit the scaler on the training rows only.
    Eigen::MatrixXd train_rows =
        feature_mat.topRows(static_cast<Eigen::Index>(train_end));

    features::StandardScaler scaler;
    auto fit_status = scaler.fit(train_rows);
    if (fit_status.is_err()) {
        return Result<PreparedData>::err(fit_status.error());
    }

    auto scaled_result = scaler.transform(feature_mat);
    if (scaled_result.is_err()) {
        return Result<PreparedData>::err(scaled_result.error());
    }
    Eigen::MatrixXd scaled = std::move(scaled_result.value());

    PreparedData prepared;
    prepared.feature_scaler = std::move(scaler);
    prepared.train =
        build_split(scaled, fm.target, 0, train_end, window_size, device);
    prepared.val =
        build_split(scaled, fm.target, train_end, val_end, window_size, device);
    prepared.test =
        build_split(scaled, fm.target, val_end, test_end, window_size, device);

    // Record raw targets/dates aligned to the test windows for reporting.
    if (val_end + static_cast<std::size_t>(window_size) < test_end) {
        for (std::size_t i = 0; i + static_cast<std::size_t>(window_size) <
                                test_end - val_end;
             ++i) {
            const std::size_t target_idx =
                val_end + i + static_cast<std::size_t>(window_size);
            prepared.test_target_raw.push_back(fm.target[target_idx]);
            prepared.test_dates.push_back(fm.dates[target_idx]);
        }
    }

    log::info(log::cat("Prepared windows -> train: ", prepared.train.size(),
                          ", val: ", prepared.val.size(),
                          ", test: ", prepared.test.size()));

    return Result<PreparedData>::ok(std::move(prepared));
}

}  // namespace trading::models
