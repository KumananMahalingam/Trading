// === FILE: cpp/src/features/scaler.cpp ===
#include "trading/features/scaler.hpp"

#include <cmath>

namespace trading::features {

Status<> StandardScaler::fit(const Eigen::MatrixXd& data) {
    if (data.rows() == 0 || data.cols() == 0) {
        return Status<>::err(
            Error(Error::Code::InvalidArgument, "Cannot fit scaler on empty matrix"));
    }

    const Eigen::Index rows = data.rows();
    const Eigen::Index cols = data.cols();

    mean_ = data.colwise().mean();

    scale_ = Eigen::VectorXd::Zero(cols);
    for (Eigen::Index c = 0; c < cols; ++c) {
        double acc = 0.0;
        for (Eigen::Index r = 0; r < rows; ++r) {
            const double d = data(r, c) - mean_(c);
            acc += d * d;
        }
        // Population variance (ddof=0), matching sklearn StandardScaler.
        double std_dev = std::sqrt(acc / static_cast<double>(rows));
        if (std_dev == 0.0 || !std::isfinite(std_dev)) {
            std_dev = 1.0;  // avoid division by zero for constant columns
        }
        scale_(c) = std_dev;
    }

    fitted_ = true;
    return ok_status();
}

Result<Eigen::MatrixXd> StandardScaler::transform(const Eigen::MatrixXd& data) const {
    if (!fitted_) {
        return Result<Eigen::MatrixXd>::err(
            Error(Error::Code::InvalidArgument, "Scaler used before fit()"));
    }
    if (data.cols() != mean_.size()) {
        return Result<Eigen::MatrixXd>::err(
            Error(Error::Code::InvalidArgument, "Column count mismatch in transform()"));
    }

    Eigen::MatrixXd out = data;
    for (Eigen::Index c = 0; c < out.cols(); ++c) {
        out.col(c) = (out.col(c).array() - mean_(c)) / scale_(c);
    }
    return Result<Eigen::MatrixXd>::ok(std::move(out));
}

Result<Eigen::MatrixXd> StandardScaler::fit_transform(const Eigen::MatrixXd& data) {
    auto status = fit(data);
    if (status.is_err()) {
        return Result<Eigen::MatrixXd>::err(status.error());
    }
    return transform(data);
}

}  // namespace trading::features
