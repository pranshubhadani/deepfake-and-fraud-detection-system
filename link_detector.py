import whois
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import datetime

def analyze_link(url):
    """
    Analyzes a given URL for common fraud/phishing indicators.
    Returns a dictionary with safety status, trust score, and reasons.
    """
    if not url.startswith('http://') and not url.startswith('https://'):
        # Default to http to see if it redirects to https or stays insecure
        url = 'http://' + url
        
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith('www.'):
        domain = domain[4:]

    suspicion_score = 0
    reasons = []

    # 1. Check Security Certificate (HTTPS)
    if parsed_url.scheme != 'https':
        suspicion_score += 2
        reasons.append("Missing security certificate (HTTP instead of HTTPS).")

    # 2. Check Link Length
    if len(url) > 75:
        suspicion_score += 1
        reasons.append("Unusually long URL, common in phishing to hide the actual domain.")

    # 3. Check for Numbers in Domain
    if re.search(r'\d', domain):
        suspicion_score += 1
        reasons.append("Domain contains numbers, which can be used to mimic real sites (e.g., g00gle.com).")

    # 4. Check for Unusual Symbols
    if '@' in url:
        suspicion_score += 3
        reasons.append("URL contains an '@' symbol, often used to obscure the true destination.")
    if domain.count('-') > 2:
        suspicion_score += 1
        reasons.append("Domain contains multiple hyphens, a common trait of fake websites.")

    # Extract a simple root domain (e.g. web.whatsapp.com -> whatsapp.com)
    domain_parts = domain.split('.')
    if len(domain_parts) > 2:
        # A rough heuristic for root domain (e.g. ignoring co.uk complexity for now)
        root_domain = '.'.join(domain_parts[-2:])
    else:
        root_domain = domain

    # 5. Check Domain Age using python-whois (with RDAP fallback)
    creation_date = None
    try:
        # WHOIS queries often fail on subdomains, so use the root_domain
        domain_info = whois.whois(root_domain)
        creation_date = domain_info.creation_date
        
        # WHOIS can sometimes return a list of dates
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
            
    except Exception:
        # Fallback to RDAP API (HTTPS) if port 43 WHOIS is blocked/rate-limited
        try:
            rdap_response = requests.get(f"https://rdap.org/domain/{root_domain}", timeout=5)
            if rdap_response.status_code == 200:
                data = rdap_response.json()
                events = data.get('events', [])
                for event in events:
                    if event.get('eventAction') == 'registration':
                        # Parse ISO date string
                        date_str = event.get('eventDate')
                        if date_str:
                            # e.g., "2005-02-15T05:13:12Z"
                            date_str = date_str.replace('Z', '+00:00')
                            creation_date = datetime.datetime.fromisoformat(date_str)
                            # Convert timezone-aware to naive for math if needed, 
                            # or just make 'now' aware. Let's make creation_date naive.
                            creation_date = creation_date.replace(tzinfo=None)
                        break
        except Exception:
            pass

    if creation_date:
        # Ensure creation_date is timezone-naive for subtraction
        creation_date = creation_date.replace(tzinfo=None)
        age_days = (datetime.datetime.now() - creation_date).days
        if age_days < 30:
            suspicion_score += 3
            reasons.append(f"Domain is extremely new ({age_days} days old). Highly suspicious.")
        elif age_days < 180:
            suspicion_score += 2
            reasons.append(f"Domain is relatively new ({age_days} days old). Proceed with caution.")
    else:
        # We don't add to suspicion_score here anymore, to avoid false positives for rate limits.
        reasons.append("Notice: Could not retrieve domain registration date (WHOIS/RDAP blocked).")

    # 6. Check Web Page Content (Headers & Outgoing Links)
    try:
        # Set a user-agent to avoid being blocked by simple bot protection
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        
        # Check Headers (e.g. strict-transport-security)
        if 'Strict-Transport-Security' not in response.headers:
            suspicion_score += 1
            reasons.append("Missing HSTS security header.")
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Pull links out of raw text
        links = soup.find_all('a', href=True)
        external_links = 0
        
        for link in links:
            href = link['href']
            # If it's an absolute URL and doesn't contain the root domain
            if href.startswith('http') and root_domain not in href:
                external_links += 1
                
        if len(links) > 0:
            external_ratio = external_links / len(links)
            if external_ratio > 0.8:
                suspicion_score += 2
                reasons.append(f"High ratio of external links ({int(external_ratio*100)}%). Typical of scam sites redirecting elsewhere.")
                
        # Look for hidden inputs which are common in phishing login pages
        hidden_inputs = soup.find_all('input', type='hidden')
        if len(hidden_inputs) > 5:
            suspicion_score += 1
            reasons.append("Suspiciously high number of hidden input fields.")
            
    except requests.exceptions.RequestException:
        suspicion_score += 2
        reasons.append("Failed to fetch page content. The site might be down, invalid, or blocking automated requests.")

    # Determine final status based on Methodology Step 5
    if suspicion_score >= 4:
        status = "Likely Fake"
    elif suspicion_score >= 2:
        status = "Needs Caution"
    else:
        status = "Safe"
        if not reasons:
            reasons.append("No obvious warning signs detected.")

    return {
        "status": status,
        "score": suspicion_score,
        "reasons": reasons
    }

def extract_links_from_text(text):
    """
    Message Cleaner: Extracts web links straight out of pasted text messages.
    Uses regex to find HTTP/HTTPS URLs or common domain patterns.
    """
    # Regex pattern to match URLs
    url_pattern = re.compile(
        r'(?:http[s]?://)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        r'(?:\.[a-zA-Z]{2,})+(?:/[a-zA-Z0-9_.-]*)*'
    )
    
    # Find all matches
    raw_links = url_pattern.findall(text)
    
    # Filter and clean
    cleaned_links = []
    for link in raw_links:
        # Avoid matching random punctuation as links
        if '.' in link and not link.endswith('.'):
            cleaned_links.append(link.strip())
            
    return list(set(cleaned_links))

if __name__ == '__main__':
    print("="*50)
    print(" 🔗 Fraud Link Detection System Ready ")
    print("="*50)
    while True:
        test_message = input("\nEnter a message or URL to verify (or type 'exit' to quit): ")
        if test_message.strip().lower() == 'exit':
            break
        if not test_message.strip():
            continue
            
        print("\nExtracting links...")
        found_links = extract_links_from_text(test_message)
        
        if not found_links:
            print("No valid links found in the message.")
            continue
            
        print(f"Found {len(found_links)} link(s):")
        for link in found_links:
            print(f"\nAnalyzing link: {link}")
            result = analyze_link(link)
            
            print("--- Verification Results ---")
            print(f"Status: {result['status']} (Score: {result['score']})")
            if result['reasons']:
                print("Reasons:")
                for r in result['reasons']:
                    print(f" - {r}")
