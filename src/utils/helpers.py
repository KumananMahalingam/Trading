"""
Utility helper functions
"""
import re
from datetime import datetime, timezone


def convert_to_iso_date(date_str):
    """Convert date string to ISO format with timezone"""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def generate_name_variations(company_name):
    """Generate variations of company names for matching"""
    variations = [company_name]

    suffixes_pattern = r'\s+(Inc\.?|Corporation|Corp\.?|Company|Co\.?|Limited|Ltd\.?|Holdings|Holding|LLC|L\.L\.C\.|plc|PLC)'
    base_name = re.sub(suffixes_pattern, '', company_name, flags=re.IGNORECASE).strip()
    variations.append(base_name)

    variations.extend([
        company_name.replace(',', '').strip(),
        company_name.replace('.', '').strip(),
        base_name.replace(',', '').strip(),
        base_name.replace('.', '').strip(),
        company_name.replace('&', 'and'),
        company_name.replace(' and ', ' & '),
        base_name.replace('&', 'and'),
        base_name.replace(' and ', ' & ')
    ])

    words = base_name.split()
    if len(words) > 1:
        acronym = ''.join([word[0] for word in words if word[0].isupper()])
        if len(acronym) >= 2:
            variations.append(acronym)

    return list(set([v.strip() for v in variations if v.strip()]))


def is_same_company(ticker1, ticker2, name1, name2, company_aliases):
    """Check if two companies are the same based on ticker and name"""
    if ticker1 == ticker2:
        return True

    for base_name, aliases in company_aliases.items():
        if ticker1 in aliases and ticker2 in aliases:
            return True

    name1_base = re.sub(r'\s+(Inc\.?|Corporation|Corp\.?|Company|Co\.?|Limited|Ltd\.?)',
                       '', name1, flags=re.IGNORECASE).strip().upper()
    name2_base = re.sub(r'\s+(Inc\.?|Corporation|Corp\.?|Company|Co\.?|Limited|Ltd\.?)',
                       '', name2, flags=re.IGNORECASE).strip().upper()

    if name1_base == name2_base:
        return True

    return False


def format_currency(value):
    """Format value as currency"""
    return f"${value:,.2f}"


def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.2f}%"


def truncate_string(text, max_length=80):
    """Truncate string with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."