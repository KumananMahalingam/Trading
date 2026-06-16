// === FILE: cpp/include/trading/features/scaler.hpp ===
#pragma once

#include <vector>

#include <Eigen/Dense>

#include "trading/util/result.hpp"

namespace trading::features {

// StandardScaler: zero mean, unit variance per column, implemented with Eigen.
// Mirrors sklearn.preprocessing.StandardScaler (population std, ddof=0) so the
// fit on training data is reused to transform validation/test data.
class StandardScaler {
public:
    StandardScaler() = default;

    // Fit on a [rows x cols] matrix, computing per-column mean and std.
    Status<> fit(const Eigen::MatrixXd& data);

    // Transform in place using previously fitted statistics.
    Result<Eigen::MatrixXd> transform(const Eigen::MatrixXd& data) const;

    // Convenience: fit then transform the same matrix.
    Result<Eigen::MatrixXd> fit_transform(const Eigen::MatrixXd& data);

    const Eigen::VectorXd& mean() const { return mean_; }
    const Eigen::VectorXd& scale() const { return scale_; }
    bool fitted() const { return fitted_; }

private:
    Eigen::VectorXd mean_;
    Eigen::VectorXd scale_;  // std, with zeros replaced by 1 to avoid div-by-0
    bool fitted_{false};
};

}  // namespace trading::features
