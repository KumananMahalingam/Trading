"""
Company name validation and matching
"""
import json
from src.utils.helpers import generate_name_variations


def load_company_tickers_json(file_path="company_tickers.json"):
    """
    Load company tickers from SEC JSON file

    Args:
        file_path: Path to company_tickers.json

    Returns:
        dict: Contains ticker_to_info, name_to_ticker, name_variations
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Loaded {len(data)} companies from {file_path}")

        ticker_to_info = {}
        name_to_ticker = {}
        name_variations = {}

        for key, company in data.items():
            ticker = company.get('ticker', '').upper()
            name = company.get('title', '')
            cik = company.get('cik_str', '')

            if ticker and name:
                ticker_to_info[ticker] = {'name': name, 'ticker': ticker, 'cik': cik}
                name_to_ticker[name.upper()] = ticker

                variations = generate_name_variations(name)
                for variation in variations:
                    if len(variation) >= 3:
                        name_variations[variation.upper()] = ticker

        print(f"Created {len(name_variations)} name variations for matching")

        return {
            'ticker_to_info': ticker_to_info,
            'name_to_ticker': name_to_ticker,
            'name_variations': name_variations
        }

    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None


def validate_company_exists(company_name, company_lookups):
    """
    Validate if a company exists in our database

    Args:
        company_name: Company name to validate
        company_lookups: Dictionary from load_company_tickers_json()

    Returns:
        tuple: (is_valid: bool, ticker: str, official_name: str)
    """
    if not company_lookups:
        return False, None, None

    company_upper = company_name.upper().strip()

    # Direct name match
    if company_upper in company_lookups['name_to_ticker']:
        ticker = company_lookups['name_to_ticker'][company_upper]
        official_name = company_lookups['ticker_to_info'][ticker]['name']
        return True, ticker, official_name

    # Name variation match
    if company_upper in company_lookups['name_variations']:
        ticker = company_lookups['name_variations'][company_upper]
        official_name = company_lookups['ticker_to_info'][ticker]['name']
        return True, ticker, official_name

    # Direct ticker match
    if company_upper in company_lookups['ticker_to_info']:
        ticker = company_upper
        official_name = company_lookups['ticker_to_info'][ticker]['name']
        return True, ticker, official_name

    # Partial match (fuzzy)
    for variation, ticker in company_lookups['name_variations'].items():
        if len(company_upper) >= 4 and company_upper in variation:
            official_name = company_lookups['ticker_to_info'][ticker]['name']
            return True, ticker, official_name
        elif len(variation) >= 4 and variation in company_upper:
            official_name = company_lookups['ticker_to_info'][ticker]['name']
            return True, ticker, official_name

    return False, None, None