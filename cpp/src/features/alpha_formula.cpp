// === FILE: cpp/src/features/alpha_formula.cpp ===
#include "trading/features/alpha_formula.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <string>

#include "trading/util/logging.hpp"

namespace trading::features {

namespace {

using VarMap = std::unordered_map<std::string, const std::vector<double>*>;

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------
struct Node {
    enum class Kind { Const, Var, Unary, Binary, Func1, Ts };
    Kind kind;
    double constant = 0.0;            // Const
    std::string name;                 // Var name / function name
    char op = 0;                      // Unary / Binary operator
    int window = 0;                   // Ts window/lag
    std::unique_ptr<Node> a;          // operand / function argument
    std::unique_ptr<Node> b;          // second operand
};
using NodePtr = std::unique_ptr<Node>;

bool is_elementwise_fn(const std::string& f) {
    return f == "abs" || f == "log" || f == "sqrt" || f == "exp" || f == "sign";
}
bool is_ts_fn(const std::string& f) {
    return f == "delay" || f == "delta" || f == "ts_mean" || f == "ts_std" ||
           f == "ts_min" || f == "ts_max";
}

// ---------------------------------------------------------------------------
// Recursive-descent parser. Returns nullptr and sets `error` on failure.
// ---------------------------------------------------------------------------
class Parser {
public:
    explicit Parser(std::string_view src) : src_(src) {}

    NodePtr parse(std::string& error) {
        auto node = parse_expr(error);
        if (!error.empty()) {
            return nullptr;
        }
        skip_ws();
        if (pos_ != src_.size()) {
            error = "Unexpected trailing characters at position " + std::to_string(pos_);
            return nullptr;
        }
        return node;
    }

private:
    void skip_ws() {
        while (pos_ < src_.size() &&
               std::isspace(static_cast<unsigned char>(src_[pos_]))) {
            ++pos_;
        }
    }

    char peek() {
        skip_ws();
        return pos_ < src_.size() ? src_[pos_] : '\0';
    }

    NodePtr parse_expr(std::string& error) {
        auto left = parse_term(error);
        if (!error.empty()) return nullptr;
        while (true) {
            const char c = peek();
            if (c == '+' || c == '-') {
                ++pos_;
                auto right = parse_term(error);
                if (!error.empty()) return nullptr;
                auto node = std::make_unique<Node>();
                node->kind = Node::Kind::Binary;
                node->op = c;
                node->a = std::move(left);
                node->b = std::move(right);
                left = std::move(node);
            } else {
                break;
            }
        }
        return left;
    }

    NodePtr parse_term(std::string& error) {
        auto left = parse_power(error);
        if (!error.empty()) return nullptr;
        while (true) {
            const char c = peek();
            if (c == '*' || c == '/') {
                ++pos_;
                auto right = parse_power(error);
                if (!error.empty()) return nullptr;
                auto node = std::make_unique<Node>();
                node->kind = Node::Kind::Binary;
                node->op = c;
                node->a = std::move(left);
                node->b = std::move(right);
                left = std::move(node);
            } else {
                break;
            }
        }
        return left;
    }

    NodePtr parse_power(std::string& error) {
        auto base = parse_unary(error);
        if (!error.empty()) return nullptr;
        if (peek() == '^') {
            ++pos_;
            auto exponent = parse_power(error);  // right associative
            if (!error.empty()) return nullptr;
            auto node = std::make_unique<Node>();
            node->kind = Node::Kind::Binary;
            node->op = '^';
            node->a = std::move(base);
            node->b = std::move(exponent);
            return node;
        }
        return base;
    }

    NodePtr parse_unary(std::string& error) {
        const char c = peek();
        if (c == '-' || c == '+') {
            ++pos_;
            auto operand = parse_unary(error);
            if (!error.empty()) return nullptr;
            if (c == '+') {
                return operand;  // unary plus is a no-op
            }
            auto node = std::make_unique<Node>();
            node->kind = Node::Kind::Unary;
            node->op = '-';
            node->a = std::move(operand);
            return node;
        }
        return parse_primary(error);
    }

    NodePtr parse_primary(std::string& error) {
        const char c = peek();
        if (c == '(') {
            ++pos_;
            auto node = parse_expr(error);
            if (!error.empty()) return nullptr;
            if (peek() != ')') {
                error = "Expected ')'";
                return nullptr;
            }
            ++pos_;
            return node;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            return parse_number(error);
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            return parse_ident(error);
        }
        error = std::string("Unexpected character '") + c + "'";
        return nullptr;
    }

    NodePtr parse_number(std::string& error) {
        skip_ws();
        const std::size_t start = pos_;
        while (pos_ < src_.size()) {
            const char c = src_[pos_];
            if (std::isdigit(static_cast<unsigned char>(c)) || c == '.' ||
                c == 'e' || c == 'E' ||
                ((c == '+' || c == '-') && pos_ > start &&
                 (src_[pos_ - 1] == 'e' || src_[pos_ - 1] == 'E'))) {
                ++pos_;
            } else {
                break;
            }
        }
        const std::string num(src_.substr(start, pos_ - start));
        try {
            auto node = std::make_unique<Node>();
            node->kind = Node::Kind::Const;
            node->constant = std::stod(num);
            return node;
        } catch (...) {
            error = "Invalid number '" + num + "'";
            return nullptr;
        }
    }

    NodePtr parse_ident(std::string& error) {
        skip_ws();
        const std::size_t start = pos_;
        while (pos_ < src_.size()) {
            const char c = src_[pos_];
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
                ++pos_;
            } else {
                break;
            }
        }
        const std::string ident(src_.substr(start, pos_ - start));

        if (peek() != '(') {
            // Plain variable reference.
            auto node = std::make_unique<Node>();
            node->kind = Node::Kind::Var;
            node->name = ident;
            return node;
        }

        // Function call.
        ++pos_;  // consume '('
        std::vector<NodePtr> args;
        if (peek() != ')') {
            while (true) {
                auto arg = parse_expr(error);
                if (!error.empty()) return nullptr;
                args.push_back(std::move(arg));
                if (peek() == ',') {
                    ++pos_;
                    continue;
                }
                break;
            }
        }
        if (peek() != ')') {
            error = "Expected ')' to close call to '" + ident + "'";
            return nullptr;
        }
        ++pos_;  // consume ')'

        if (is_elementwise_fn(ident)) {
            if (args.size() != 1) {
                error = ident + "() expects 1 argument";
                return nullptr;
            }
            auto node = std::make_unique<Node>();
            node->kind = Node::Kind::Func1;
            node->name = ident;
            node->a = std::move(args[0]);
            return node;
        }
        if (is_ts_fn(ident)) {
            if (args.size() != 2 || args[1]->kind != Node::Kind::Const) {
                error = ident + "(x, n) expects a series and an integer window";
                return nullptr;
            }
            auto node = std::make_unique<Node>();
            node->kind = Node::Kind::Ts;
            node->name = ident;
            node->a = std::move(args[0]);
            node->window = std::max(1, static_cast<int>(args[1]->constant));
            return node;
        }
        error = "Unknown function '" + ident + "'";
        return nullptr;
    }

    std::string_view src_;
    std::size_t pos_ = 0;
};

// Collect variable names referenced by the AST.
void collect_vars(const Node& node, std::set<std::string>& out) {
    switch (node.kind) {
        case Node::Kind::Var:
            out.insert(node.name);
            break;
        case Node::Kind::Const:
            break;
        case Node::Kind::Unary:
        case Node::Kind::Func1:
        case Node::Kind::Ts:
            if (node.a) collect_vars(*node.a, out);
            break;
        case Node::Kind::Binary:
            if (node.a) collect_vars(*node.a, out);
            if (node.b) collect_vars(*node.b, out);
            break;
    }
}

// ---------------------------------------------------------------------------
// Evaluation (element-wise, producing a length-n vector). All variables and
// functions are validated before eval, so this cannot fail.
// ---------------------------------------------------------------------------
std::vector<double> eval(const Node& node, const VarMap& vars, std::size_t n);

std::vector<double> eval_ts(const std::string& fn, const std::vector<double>& x,
                            int k) {
    const std::size_t n = x.size();
    std::vector<double> out(n, 0.0);
    auto delay = [&](std::size_t i) -> double {
        return x[(i >= static_cast<std::size_t>(k)) ? i - static_cast<std::size_t>(k) : 0];
    };
    for (std::size_t i = 0; i < n; ++i) {
        if (fn == "delay") {
            out[i] = delay(i);
            continue;
        }
        if (fn == "delta") {
            out[i] = x[i] - delay(i);
            continue;
        }
        const std::size_t begin =
            (i + 1 >= static_cast<std::size_t>(k)) ? i + 1 - static_cast<std::size_t>(k) : 0;
        const std::size_t count = i - begin + 1;
        if (fn == "ts_mean" || fn == "ts_std") {
            double sum = 0.0;
            for (std::size_t j = begin; j <= i; ++j) sum += x[j];
            const double mean = sum / static_cast<double>(count);
            if (fn == "ts_mean") {
                out[i] = mean;
            } else {
                double var = 0.0;
                for (std::size_t j = begin; j <= i; ++j) {
                    const double d = x[j] - mean;
                    var += d * d;
                }
                out[i] = std::sqrt(var / static_cast<double>(count));
            }
        } else if (fn == "ts_min") {
            double m = x[begin];
            for (std::size_t j = begin; j <= i; ++j) m = std::min(m, x[j]);
            out[i] = m;
        } else if (fn == "ts_max") {
            double m = x[begin];
            for (std::size_t j = begin; j <= i; ++j) m = std::max(m, x[j]);
            out[i] = m;
        }
    }
    return out;
}

std::vector<double> eval(const Node& node, const VarMap& vars, std::size_t n) {
    switch (node.kind) {
        case Node::Kind::Const:
            return std::vector<double>(n, node.constant);
        case Node::Kind::Var: {
            const auto it = vars.find(node.name);
            return *it->second;  // validated to exist
        }
        case Node::Kind::Unary: {
            auto a = eval(*node.a, vars, n);
            for (auto& v : a) v = -v;
            return a;
        }
        case Node::Kind::Func1: {
            auto a = eval(*node.a, vars, n);
            for (auto& v : a) {
                if (node.name == "abs") v = std::abs(v);
                else if (node.name == "log") v = (v > 0.0) ? std::log(v) : 0.0;
                else if (node.name == "sqrt") v = (v >= 0.0) ? std::sqrt(v) : 0.0;
                else if (node.name == "exp") v = std::exp(v);
                else if (node.name == "sign") v = (v > 0.0) ? 1.0 : (v < 0.0 ? -1.0 : 0.0);
            }
            return a;
        }
        case Node::Kind::Ts: {
            auto a = eval(*node.a, vars, n);
            return eval_ts(node.name, a, node.window);
        }
        case Node::Kind::Binary: {
            auto a = eval(*node.a, vars, n);
            auto b = eval(*node.b, vars, n);
            for (std::size_t i = 0; i < n; ++i) {
                switch (node.op) {
                    case '+': a[i] = a[i] + b[i]; break;
                    case '-': a[i] = a[i] - b[i]; break;
                    case '*': a[i] = a[i] * b[i]; break;
                    case '/': a[i] = (b[i] != 0.0) ? a[i] / b[i] : 0.0; break;
                    case '^': a[i] = std::pow(a[i], b[i]); break;
                    default: break;
                }
            }
            return a;
        }
    }
    return std::vector<double>(n, 0.0);
}

// Replace non-finite values with 0 and clip to mean +/- 3*std (population).
void sanitize_and_clip(std::vector<double>& col) {
    for (auto& v : col) {
        if (!std::isfinite(v)) v = 0.0;
    }
    if (col.empty()) return;
    double sum = 0.0;
    for (double v : col) sum += v;
    const double mean = sum / static_cast<double>(col.size());
    double var = 0.0;
    for (double v : col) {
        const double d = v - mean;
        var += d * d;
    }
    const double sd = std::sqrt(var / static_cast<double>(col.size()));
    if (sd > 0.0) {
        const double lo = mean - 3.0 * sd;
        const double hi = mean + 3.0 * sd;
        for (auto& v : col) v = std::min(hi, std::max(lo, v));
    }
}

// Replace a UTF-8 / ASCII substring with another, in place.
void replace_all(std::string& s, std::string_view from, std::string_view to) {
    if (from.empty()) return;
    std::size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string trim(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n\"'");
    if (begin == std::string::npos) return "";
    const auto end = s.find_last_not_of(" \t\r\n\"'");
    return s.substr(begin, end - begin + 1);
}

}  // namespace

Result<std::vector<double>> evaluate_formula(
    std::string_view formula, const VarMap& variables, std::size_t n) {
    Parser parser(formula);
    std::string error;
    NodePtr ast = parser.parse(error);
    if (!error.empty() || !ast) {
        return Result<std::vector<double>>::err(
            Error(Error::Code::Parse,
                  "Formula parse error: " + error + " in '" + std::string(formula) + "'"));
    }

    std::set<std::string> used;
    collect_vars(*ast, used);
    for (const auto& name : used) {
        if (variables.find(name) == variables.end()) {
            return Result<std::vector<double>>::err(
                Error(Error::Code::InvalidArgument,
                      "Unknown variable '" + name + "' in formula '" +
                          std::string(formula) + "'"));
        }
    }

    return Result<std::vector<double>>::ok(eval(*ast, variables, n));
}

std::vector<std::string> parse_formula_lines(std::string_view text) {
    std::vector<std::string> formulas;
    std::string buffer(text);

    // Normalize unicode operators and Python-isms.
    replace_all(buffer, "\xC3\x97", "*");      // × multiplication sign
    replace_all(buffer, "\xC3\xB7", "/");      // ÷ division sign
    replace_all(buffer, "\xE2\x88\x92", "-");  // − minus sign
    replace_all(buffer, "\xE2\x80\x93", "-");  // – en dash
    replace_all(buffer, "**", "^");            // Python power -> our '^'
    replace_all(buffer, "np.", "");            // strip numpy namespace

    std::size_t start = 0;
    while (start <= buffer.size()) {
        std::size_t nl = buffer.find('\n', start);
        if (nl == std::string::npos) nl = buffer.size();
        std::string line = buffer.substr(start, nl - start);
        start = nl + 1;

        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string rhs = trim(line.substr(eq + 1));
        if (rhs.size() >= 3) {
            formulas.push_back(rhs);
        }
    }
    return formulas;
}

Status<> append_alphas(FeatureMatrix& fm,
                       const std::vector<std::string>& formulas,
                       int max_alphas) {
    if (fm.empty()) {
        return Status<>::err(Error(Error::Code::InvalidArgument, "Empty feature matrix"));
    }

    const std::size_t n = fm.size();
    const int base_cols = fm.num_features();

    // Extract base indicator columns into contiguous series for evaluation.
    std::unordered_map<std::string, std::vector<double>> series;
    for (int c = 0; c < base_cols; ++c) {
        std::vector<double> col(n);
        for (std::size_t r = 0; r < n; ++r) {
            col[r] = fm.features[r][static_cast<std::size_t>(c)];
        }
        series[fm.feature_names[static_cast<std::size_t>(c)]] = std::move(col);
    }
    VarMap vars;
    for (const auto& kv : series) {
        vars[kv.first] = &kv.second;
    }

    int added = 0;
    for (const auto& formula : formulas) {
        if (added >= max_alphas) break;
        auto result = evaluate_formula(formula, vars, n);
        if (result.is_err()) {
            log::warn(log::cat("Skipping alpha (", result.error().message, ")"));
            continue;
        }
        std::vector<double> col = std::move(result.value());
        sanitize_and_clip(col);

        ++added;
        const std::string col_name = "alpha_" + std::to_string(added);
        fm.feature_names.push_back(col_name);
        for (std::size_t r = 0; r < n; ++r) {
            fm.features[r].push_back(col[r]);
        }
        log::info(log::cat("  + ", col_name, " = ", formula));
    }

    if (added == 0) {
        return Status<>::err(
            Error(Error::Code::InvalidArgument, "No alpha formulas evaluated successfully"));
    }
    return ok_status();
}

}  // namespace trading::features
