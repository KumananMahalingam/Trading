// === FILE: cpp/include/trading/storage/excel_handler.hpp ===
#pragma once

#include <string>
#include <string_view>

#include "trading/features/technical_indicators.hpp"
#include "trading/models/trainer.hpp"
#include "trading/util/result.hpp"

namespace trading::storage {

// Write the computed 5-feature matrix (with dates, close, target) to an .xlsx
// workbook using xlnt.
Status<> save_features_to_excel(std::string_view path,
                                const features::FeatureMatrix& fm);

// Write test-set predictions vs. actuals (with uncertainty) and a metrics
// summary sheet to an .xlsx workbook.
Status<> save_results_to_excel(std::string_view path,
                               std::string_view ticker,
                               const models::TrainingResult& result);

}  // namespace trading::storage
