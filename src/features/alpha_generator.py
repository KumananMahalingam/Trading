"""
Alpha formula generation using Groq LLM
"""
import time


def generate_simple_alphas(ticker):
    """
    Fallback alphas that don't require related company sentiment

    Args:
        ticker: Stock ticker symbol

    Returns:
        str: Simple alpha formulas
    """
    return f"""
α1 = Return_5D
α2 = (RSI - 50) / 50
α3 = MACD - MACD_Signal
α4 = (close - SMA_20) / SMA_20
α5 = Return_5D + 0.1 * {ticker}_Sentiment
"""


def generate_alphas_with_groq(groq_client, ticker, company_name, comprehensive_df,
                              related_companies, max_retries=3):
    """
    Generate predictive alphas using Groq LLM with enhanced prompt

    Args:
        groq_client: Groq API client
        ticker: Stock ticker symbol
        company_name: Company name
        comprehensive_df: DataFrame with all features
        related_companies: List of related tickers
        max_retries: Maximum retry attempts

    Returns:
        str: Generated alpha formulas
    """
    print(f"\n  Generating alphas for {company_name} ({ticker}) using Groq LLM...")

    cutoff_idx = int(len(comprehensive_df) * 0.8)
    sample_df = comprehensive_df.iloc[:cutoff_idx].tail(20).copy()

    # Select only most important columns to reduce token usage
    important_cols = ['date', 'close', 'volume']

    tech_cols = [col for col in sample_df.columns if any(ind in col for ind in
                ['RSI', 'MACD', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'ATR',
                 'BB_Upper', 'BB_Lower', 'Return_5D', 'Return_20D', 'Volatility_20D', 'Volume_Ratio'])]
    important_cols.extend(tech_cols)

    sentiment_cols = [col for col in sample_df.columns if 'Sentiment' in col]
    important_cols.extend(sentiment_cols)

    econ_cols = [col for col in sample_df.columns if any(ind in col for ind in
                ['VIX', 'Fed_Funds_Rate', 'Treasury_10Y', 'CPI', 'Unemployment_Rate'])]
    important_cols.extend(econ_cols)

    alt_cols = [col for col in sample_df.columns if any(ind in col for ind in
               ['SEC_Sentiment', 'Earnings', 'Days_Since_Earnings'])]
    important_cols.extend(alt_cols)

    important_cols = list(dict.fromkeys(important_cols))
    sample_df = sample_df[[col for col in important_cols if col in sample_df.columns]]

    available_features = list(sample_df.columns)

    # Categorize features
    stock_features = [col for col in available_features if col in ['close', 'open', 'high', 'low', 'volume', 'date']]

    technical_indicators = [col for col in available_features if any(ind in col.upper() for ind in
                           ['SMA', 'EMA', 'RSI', 'MACD', 'BB_', 'ATR', 'OBV', 'MFI', 'STOCHASTIC',
                            'WILLIAMS', 'ADX', 'ROC', 'MOMENTUM', 'VOLATILITY', 'RETURN'])]

    sentiment_features = [col for col in available_features if 'Sentiment' in col]

    target_sentiment = [col for col in sentiment_features if col.startswith(f'{ticker}_')]
    related_sentiment = [col for col in sentiment_features if not col.startswith(f'{ticker}_') and not col.startswith('Sentiment_Div')]
    sentiment_divergence = [col for col in sentiment_features if col.startswith('Sentiment_Div')]

    economic_features = [col for col in available_features if any(ind in col for ind in
                        ['GDP', 'CPI', 'Unemployment', 'Fed_Funds', 'Treasury', 'VIX', 'Oil_Price',
                         'Gold_Price', 'Consumer_Sentiment', 'Retail_Sales'])]

    alternative_features = [col for col in available_features if any(ind in col for ind in
                           ['SEC', 'Earnings', 'Days_Since_Earnings'])]

    target_col = ticker + '_Sentiment'
    stats_context = ""
    if target_col in sample_df.columns:
        sent_mean = sample_df[target_col].mean()
        sent_std = sample_df[target_col].std()
        sent_min = sample_df[target_col].min()
        sent_max = sample_df[target_col].max()

        vol_20d_str = f"{sample_df['Volatility_20D'].iloc[-1]:.4f}" if 'Volatility_20D' in sample_df.columns else 'N/A'
        ret_20d_str = f"{sample_df['Return_20D'].iloc[-1]:.4f}" if 'Return_20D' in sample_df.columns else 'N/A'

        stats_context = f"""
Statistical Context:
- {ticker} Sentiment: Mean={sent_mean:.3f}, Std={sent_std:.3f}, Range=[{sent_min:.3f}, {sent_max:.3f}]
- Recent volatility (20D): {vol_20d_str}
- Recent return (20D): {ret_20d_str}
"""

    related_str = "None available"
    if related_companies:
        related_with_overlap = []
        for rel_ticker in related_companies:
            if f'{rel_ticker}_Sentiment' in sample_df.columns:
                related_with_overlap.append(f"{rel_ticker} (sentiment available)")
            else:
                related_with_overlap.append(f"{rel_ticker} (no sentiment)")
        related_str = ", ".join(related_with_overlap)

    df_summary = f"""
Date Range: {sample_df['date'].iloc[0]} to {sample_df['date'].iloc[-1]} ({len(sample_df)} days)

Recent Values (last 5 days):
{sample_df.tail(5).to_string(index=False, max_cols=10)}

Feature Statistics:
{sample_df.describe().loc[['mean', 'std', 'min', 'max']].to_string()}
"""

    example_alpha_1 = f"Sentiment_Div_{ticker}_MSFT / (Volatility_20D + 1e-8)"
    example_alpha_2 = f"(RSI - 50) / 50 + 0.3 * {ticker}_Sentiment_MA5"
    example_alpha_3 = f"(MACD_Hist / ATR) * (1 + {ticker}_Sentiment_Change)"
    example_alpha_4 = f"(SMA_5 - SMA_50) / close * (1 - Unemployment_Rate / 10)"
    example_alpha_5 = f"Return_20D * Volume_Ratio * (1 + {ticker}_Days_Since_Earnings / 90)"

    sentiment_example_features = f"{ticker}_Sentiment_MA5, {ticker}_Sentiment_Lag5"

    prompt = f"""You are a quantitative finance expert designing predictive alpha factors for algorithmic trading.

TASK: Generate 5 formulaic alpha signals to predict {company_name} ({ticker}) stock price movements.

DATA AVAILABLE:
════════════════════════════════════════════════════════════════════════════════
1. STOCK FEATURES ({len(stock_features)}):
   {', '.join(stock_features)}

2. TECHNICAL INDICATORS ({len(technical_indicators)}):
   Available: RSI, MACD, SMA (5/10/20/50/200), EMA (12/26), BB, ATR, Volume_Ratio, Returns, Volatility

3. SENTIMENT FEATURES ({len(sentiment_features)}):
   Target: {', '.join(target_sentiment[:5])}
   Related: {', '.join(related_sentiment[:5])}{'...' if len(related_sentiment) > 5 else ''}
   Divergences: {len(sentiment_divergence)} available

4. ECONOMIC INDICATORS ({len(economic_features)}):
   Available: {', '.join(economic_features[:10])}{'...' if len(economic_features) > 10 else ''}

5. ALTERNATIVE DATA ({len(alternative_features)}):
   {', '.join(alternative_features)}

RELATED COMPANIES: {related_str}

{stats_context}

SAMPLE DATA (Last 20 trading days - summary):
{df_summary}

ALPHA GENERATION REQUIREMENTS:
════════════════════════════════════════════════════════════════════════════════

Generate exactly 5 diverse alpha formulas following these guidelines:

1. UTILIZE MULTIPLE DATA SOURCES:
   - α1: Focus on CROSS-COMPANY SENTIMENT DIVERGENCE (use Sentiment_Div features or create divergences)
   - α2: Combine TECHNICAL INDICATORS with TARGET SENTIMENT
   - α3: Use MOMENTUM + VOLATILITY signals (trend-following)
   - α4: Incorporate ECONOMIC INDICATORS + stock technicals
   - α5: Alternative data fusion (SEC sentiment, earnings events, or multi-timeframe technical)

2. FEATURE ENGINEERING BEST PRACTICES:
   ✓ Normalize by volatility (divide by ATR or Volatility_20D)
   ✓ Use ratio features (e.g., Volume_Ratio, BB_Position)
   ✓ Combine short-term and long-term signals (e.g., SMA_5 - SMA_50)
   ✓ Leverage sentiment lags and moving averages (e.g., {sentiment_example_features})
   ✓ Create divergence signals between related metrics

3. MATHEMATICAL VALIDITY:
   ✓ Avoid division by zero (use ATR, std, or add small epsilon: 1e-8)
   ✓ Keep formulas executable (2-6 terms per alpha)
   ✓ Use ONLY column names from the data above
   ✓ Ensure formulas return numeric values

4. STRATEGIC DIVERSITY:
   - Include at least ONE mean-reversion alpha (e.g., Bollinger Band position)
   - Include at least ONE momentum/trend alpha (e.g., MACD, ROC)
   - Include at least ONE sentiment-based alpha
   - Avoid redundant alphas (don't repeat similar logic)

OUTPUT FORMAT (respond with ONLY these 5 lines, no explanations):
α1 = <formula>
α2 = <formula>
α3 = <formula>
α4 = <formula>
α5 = <formula>

EXAMPLE ALPHAS (for reference only - generate NEW ones based on actual data):
α1 = {example_alpha_1}
α2 = {example_alpha_2}
α3 = {example_alpha_3}
α4 = {example_alpha_4}
α5 = {example_alpha_5}

NOW GENERATE 5 UNIQUE ALPHAS FOR {company_name}:"""

    for attempt in range(max_retries):
        try:
            print(f"    Attempt {attempt + 1}/{max_retries}...")

            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a quantitative finance expert specializing in alpha generation. Output ONLY the 5 alpha formulas in the exact format specified. No preamble, no explanations, no markdown formatting."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.8,
                max_tokens=1500,
                top_p=0.95
            )

            alpha_text = response.choices[0].message.content.strip()

            # Extract alpha lines
            alpha_lines = [line for line in alpha_text.split('\n') if line.strip().startswith('α')]

            if len(alpha_lines) >= 5:
                print(f"    ✓ Successfully generated {len(alpha_lines)} alphas")

                all_alphas_text = '\n'.join(alpha_lines)
                features_used = set()

                for feature_group in [sentiment_divergence, target_sentiment,
                                     technical_indicators[:20], economic_features[:10]]:
                    for feature in feature_group:
                        if feature in all_alphas_text:
                            features_used.add(feature)

                print(f"    ✓ Alphas utilize {len(features_used)} unique features")

                if any('Sentiment_Div' in line for line in alpha_lines):
                    print(f"    ✓ Sentiment divergence signals included")

                return alpha_text
            else:
                print(f"    ⚠ Only got {len(alpha_lines)} alphas, retrying...")

        except Exception as e:
            print(f"    ✗ Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"    Falling back to simple alphas for {ticker}")
                return generate_simple_alphas(ticker)

    return generate_simple_alphas(ticker)