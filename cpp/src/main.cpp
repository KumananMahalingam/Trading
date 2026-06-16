// === FILE: cpp/src/main.cpp ===
#include <string>
#include <string_view>

#include <torch/torch.h>

#include "trading/config/settings.hpp"
#include "trading/data/yahoo_client.hpp"
#include "trading/features/alpha_formula.hpp"
#include "trading/features/alpha_generator.hpp"
#include "trading/features/technical_indicators.hpp"
#include "trading/models/dataset.hpp"
#include "trading/models/return_predictor.hpp"
#include "trading/models/trainer.hpp"
#include "trading/storage/excel_handler.hpp"
#include "trading/util/logging.hpp"

namespace {

namespace log = trading::log;

struct CliArgs {
    std::string ticker{std::string(trading::config::kDefaultTicker)};
    std::string start_date{std::string(trading::config::kDefaultStartDate)};
    std::string end_date{std::string(trading::config::kDefaultEndDate)};
};

CliArgs parse_args(int argc, char** argv) {
    CliArgs args;
    if (argc > 1) args.ticker = argv[1];
    if (argc > 2) args.start_date = argv[2];
    if (argc > 3) args.end_date = argv[3];
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    log::init();
    const CliArgs args = parse_args(argc, argv);

    log::info("==== LSTM Trading Platform (C++ / 5 features) ====");
    log::info(log::cat("Ticker: ", args.ticker, "  Range: ", args.start_date,
                       " -> ", args.end_date));

    // 1) Fetch OHLCV.
    trading::data::YahooClient client;
    auto series_res = client.fetch_daily(args.ticker, args.start_date, args.end_date);
    if (series_res.is_err()) {
        log::error(log::cat("[", trading::to_string(series_res.error().code), "] ",
                            series_res.error().message));
        return 1;
    }
    const auto& series = series_res.value();

    // 2) Compute the 5 features + next-day-return target.
    auto fm_res = trading::features::build_feature_matrix(series);
    if (fm_res.is_err()) {
        log::error(log::cat("[", trading::to_string(fm_res.error().code), "] ",
                            fm_res.error().message));
        return 1;
    }
    auto& fm = fm_res.value();

    // 2b) Generate and append LLM-derived (or fallback) alpha signals computed
    //     from the 5 base indicators.
    if (trading::config::kUseAlphas) {
        auto formulas = trading::features::obtain_alpha_formulas(args.ticker);
        if (auto st = trading::features::append_alphas(fm, formulas,
                                                       trading::config::kNumAlphas);
            st.is_err()) {
            log::warn(log::cat("Alpha generation skipped: ", st.error().message));
        }
    }

    log::info(log::cat("Feature matrix: ", fm.size(), " rows x ",
                       fm.num_features(), " features"));

    if (auto st = trading::storage::save_features_to_excel(
            args.ticker + "_features.xlsx", fm);
        st.is_err()) {
        log::warn(log::cat("Could not save features: ", st.error().message));
    }

    // 3) Window + scale + split.
    torch::manual_seed(42);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    log::info(log::cat("Using device: ", device.is_cuda() ? "CUDA" : "CPU"));

    auto data_res = trading::models::prepare_data(fm, trading::config::kWindowSize, device);
    if (data_res.is_err()) {
        log::error(log::cat("[", trading::to_string(data_res.error().code), "] ",
                            data_res.error().message));
        return 1;
    }
    const auto& data = data_res.value();

    // 4) Build the model (input width = base indicators + appended alphas).
    trading::models::ReturnPredictor model(
        fm.num_features(), trading::config::kHiddenSize,
        trading::config::kNumLayers, trading::config::kDropout,
        trading::config::kNumHeads);

    int64_t num_params = 0;
    for (const auto& p : model->parameters()) {
        num_params += p.numel();
    }
    log::info(log::cat("Model parameters: ", num_params));

    // 5) Train + evaluate.
    auto train_res = trading::models::train_and_evaluate(model, data, device);
    if (train_res.is_err()) {
        log::error(log::cat("[", trading::to_string(train_res.error().code), "] ",
                            train_res.error().message));
        return 1;
    }
    const auto& result = train_res.value();
    const auto& m = result.metrics;

    // 6) Report.
    log::info(log::cat("==== Test Set Performance (", args.ticker, ") ===="));
    log::info(log::cat("  RMSE                 : ", log::fixed(m.rmse, 6)));
    log::info(log::cat("  MAE                  : ", log::fixed(m.mae, 6)));
    log::info(log::cat("  Directional Accuracy : ",
                       log::fixed(m.directional_accuracy, 2), "%"));
    log::info(log::cat("  Sharpe Ratio         : ", log::fixed(m.sharpe_ratio, 3)));
    log::info(log::cat("  Win Rate             : ", log::fixed(m.win_rate, 2), "%"));
    log::info(log::cat("  Avg Uncertainty      : ", log::fixed(m.avg_uncertainty, 6)));

    if (auto st = trading::storage::save_results_to_excel(
            args.ticker + "_results.xlsx", args.ticker, result);
        st.is_err()) {
        log::warn(log::cat("Could not save results: ", st.error().message));
    }

    // 7) Persist the trained model.
    try {
        torch::save(model, args.ticker + "_model.pt");
        log::info(log::cat("Saved model to ", args.ticker, "_model.pt"));
    } catch (const std::exception& e) {
        log::warn(log::cat("Could not save model: ", e.what()));
    }

    log::info("Done.");
    return 0;
}
