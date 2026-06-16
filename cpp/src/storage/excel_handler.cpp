#include "trading/util/logging.hpp"
// === FILE: cpp/src/storage/excel_handler.cpp ===
#include "trading/storage/excel_handler.hpp"

#include <string>

#include <xlnt/xlnt.hpp>

#include "trading/config/settings.hpp"

namespace trading::storage {

Status<> save_features_to_excel(std::string_view path,
                                const features::FeatureMatrix& fm) {
    if (fm.empty()) {
        return Status<>::err(
            Error(Error::Code::InvalidArgument, "Feature matrix is empty"));
    }

    try {
        xlnt::workbook wb;
        xlnt::worksheet ws = wb.active_sheet();
        ws.title("features");

        // Header row.
        ws.cell(1, 1).value("date");
        ws.cell(2, 1).value("close");
        const int nf = fm.num_features();
        for (int f = 0; f < nf; ++f) {
            ws.cell(static_cast<xlnt::column_t>(3 + f), 1)
                .value(fm.feature_names[static_cast<std::size_t>(f)]);
        }
        ws.cell(static_cast<xlnt::column_t>(3 + nf), 1).value("target_next_return");

        // Data rows (xlnt is 1-indexed; column, row).
        for (std::size_t r = 0; r < fm.size(); ++r) {
            const auto row = static_cast<xlnt::row_t>(r + 2);
            ws.cell(1, row).value(fm.dates[r]);
            ws.cell(2, row).value(fm.close[r]);
            for (int f = 0; f < nf; ++f) {
                ws.cell(static_cast<xlnt::column_t>(3 + f), row)
                    .value(fm.features[r][static_cast<std::size_t>(f)]);
            }
            ws.cell(static_cast<xlnt::column_t>(3 + nf), row)
                .value(fm.target[r]);
        }

        wb.save(std::string(path));
    } catch (const std::exception& e) {
        return Status<>::err(
            Error(Error::Code::Io, std::string("xlnt save failed: ") + e.what()));
    }

    log::info(log::cat("Saved features to ", std::string(path)));
    return ok_status();
}

Status<> save_results_to_excel(std::string_view path,
                               std::string_view ticker,
                               const models::TrainingResult& result) {
    try {
        xlnt::workbook wb;

        // Sheet 1: predictions.
        xlnt::worksheet preds = wb.active_sheet();
        preds.title("predictions");
        preds.cell(1, 1).value("date");
        preds.cell(2, 1).value("actual_return");
        preds.cell(3, 1).value("predicted_return");
        preds.cell(4, 1).value("uncertainty");

        const std::size_t n = result.predictions.size();
        for (std::size_t i = 0; i < n; ++i) {
            const auto row = static_cast<xlnt::row_t>(i + 2);
            if (i < result.dates.size()) {
                preds.cell(1, row).value(result.dates[i]);
            }
            preds.cell(2, row).value(result.actuals[i]);
            preds.cell(3, row).value(result.predictions[i]);
            if (i < result.uncertainties.size()) {
                preds.cell(4, row).value(result.uncertainties[i]);
            }
        }

        // Sheet 2: metrics summary.
        xlnt::worksheet summary = wb.create_sheet();
        summary.title("metrics");
        const auto& m = result.metrics;
        int r = 1;
        auto put = [&](const std::string& k, double v) {
            summary.cell(1, static_cast<xlnt::row_t>(r)).value(k);
            summary.cell(2, static_cast<xlnt::row_t>(r)).value(v);
            ++r;
        };
        summary.cell(1, 1).value("ticker");
        summary.cell(2, 1).value(std::string(ticker));
        ++r;
        put("RMSE", m.rmse);
        put("MAE", m.mae);
        put("Directional Accuracy (%)", m.directional_accuracy);
        put("Sharpe Ratio", m.sharpe_ratio);
        put("Win Rate (%)", m.win_rate);
        put("Avg Uncertainty", m.avg_uncertainty);
        put("Best Epoch", static_cast<double>(result.best_epoch + 1));
        put("Best Val Loss", result.best_val_loss);

        wb.save(std::string(path));
    } catch (const std::exception& e) {
        return Status<>::err(
            Error(Error::Code::Io, std::string("xlnt save failed: ") + e.what()));
    }

    log::info(log::cat("Saved results to ", std::string(path)));
    return ok_status();
}

}  // namespace trading::storage
