"""
Generate data/real_merchants.csv for SmartSpend.

Each row is one bank-statement style description with the category it belongs to.
Columns:
  description : the text a bank would show (this is what the model reads)
  category    : one of the team categories (the label the model learns)
  brand       : the merchant behind the row (use this to split train/test by brand)

How to extend it (anyone on the team can do this, no coding needed):
  1. Find the category in BRANDS below.
  2. Add the merchant name inside the quotes, separated by commas.
  3. Run:  py src/generate_merchants.py
  4. Commit data/real_merchants.csv

Run from the main smartspend folder so the file lands in data/.
"""

import csv
import random
import string

rng = random.Random(42)  # fixed seed so everyone gets the same file

# ---------------------------------------------------------------------------
# 1. Real merchants per category. Edit these lists to add more.
# ---------------------------------------------------------------------------
BRANDS = {
    "Groceries": [
        "Publix", "Walmart", "Winn-Dixie", "Aldi", "Trader Joe's", "Whole Foods",
        "Sam's Club", "Costco", "Sprouts", "Lidl", "Save A Lot", "Fresh Market",
        "Amazon Fresh", "Instacart", "Gopuff", "Harris Teeter", "BJ's Wholesale",
    ],
    "Dining": [
        "Taco Bell", "McDonald's", "Chipotle", "Chick-fil-A", "Starbucks", "Wawa",
        "Dunkin", "Panda Express", "Subway", "Wendy's", "Burger King", "Domino's",
        "Papa John's", "Pizza Hut", "Little Caesars", "Jimmy John's", "Zaxby's",
        "Popeyes", "KFC", "Raising Cane's", "Five Guys", "Shake Shack", "Panera Bread",
        "Moe's Southwest Grill", "Tijuana Flats", "Qdoba", "Sonic Drive-In", "Arby's",
        "Jersey Mike's", "Firehouse Subs", "Wingstop", "Buffalo Wild Wings",
        "Texas Roadhouse", "Olive Garden", "Applebee's", "IHOP", "Waffle House",
        "Cracker Barrel", "Chili's", "Outback Steakhouse", "Hooters", "Smoothie King",
        "Tropical Smoothie Cafe", "Krispy Kreme", "Insomnia Cookies", "Dairy Queen",
        "Cold Stone Creamery", "Baskin-Robbins", "Blaze Pizza", "Cava", "Sweetgreen",
        "Marco's Pizza", "Pollo Tropical", "DoorDash", "Uber Eats", "Grubhub",
        "Gator Dining",
    ],
    "Transportation": [
        "Uber", "Lyft", "Shell", "Exxon", "BP", "Chevron", "RaceTrac", "Circle K",
        "Sunoco", "Mobil", "Marathon", "Texaco", "Valero", "Murphy USA",
        "Spirit Airlines", "Delta Air Lines", "Southwest Airlines", "JetBlue",
        "American Airlines", "United Airlines", "Frontier Airlines", "Amtrak",
        "Greyhound", "FlixBus", "SunPass", "Enterprise Rent-A-Car", "Hertz", "Avis",
        "Zipcar", "Bird", "Lime", "Jiffy Lube", "AutoZone", "Advance Auto Parts",
        "O'Reilly Auto Parts", "Valvoline", "Firestone", "Pep Boys", "UF Parking",
        "Gainesville RTS", "Airbnb", "Expedia", "Booking.com", "Marriott", "Hilton",
        "Hampton Inn", "Holiday Inn",
    ],
    "Shopping": [
        "Amazon", "Target", "Nike", "Best Buy", "Apple Store", "Shein", "Temu", "eBay",
        "Etsy", "Old Navy", "H&M", "Zara", "Urban Outfitters", "Lululemon", "Adidas",
        "Foot Locker", "Dick's Sporting Goods", "Academy Sports", "Ross", "TJ Maxx",
        "Marshalls", "Home Depot", "Lowe's", "IKEA", "Dollar Tree", "Dollar General",
        "Five Below", "Michaels", "Hobby Lobby", "GameStop", "Barnes & Noble",
        "Wayfair", "Macy's", "Kohl's", "Nordstrom Rack", "Forever 21", "American Eagle",
        "Hollister", "Gap", "Victoria's Secret", "Micro Center", "Staples",
        "Office Depot", "Goodwill", "Poshmark", "Depop", "Mercari", "Vans", "Converse",
        "Champs Sports", "Journeys", "Fanatics",
    ],
    "Entertainment": [
        "Netflix", "Spotify", "Hulu", "Disney Plus", "HBO Max", "Peacock",
        "Paramount Plus", "YouTube Premium", "Apple Music", "Apple TV",
        "Amazon Prime Video", "Audible", "AMC Theatres", "Regal Cinemas", "Cinemark",
        "Steam", "PlayStation Network", "Xbox", "Nintendo", "Twitch", "Ticketmaster",
        "StubHub", "SeatGeek", "Live Nation", "Universal Studios", "Walt Disney World",
        "Dave & Buster's", "Topgolf", "Bowlero", "Crunchyroll", "Fandango",
        "Epic Games", "Roblox", "Discord Nitro", "SiriusXM", "Pandora", "UF Athletics",
        "Busch Gardens", "SeaWorld",
    ],
    "Housing": [
        "Gainesville Regional Utilities", "Duke Energy", "FPL", "Spectrum", "Xfinity",
        "Comcast", "AT&T Internet", "Lemonade", "Water Bill", "Electric Bill",
        "RentCafe", "AppFolio", "Zillow Rentals", "Apartments.com",
    ],
    "Healthcare": [
        "CVS Pharmacy", "Walgreens", "Rite Aid", "Publix Pharmacy", "Walmart Pharmacy",
        "UF Health", "UF Health Shands", "UF Student Health Care Center",
        "HCA Florida North Florida Hospital", "AdventHealth", "Quest Diagnostics",
        "Labcorp", "MinuteClinic", "Zocdoc", "Hims", "GoodRx", "LensCrafters",
        "Warby Parker", "Aetna", "Blue Cross Blue Shield", "Cigna", "UnitedHealthcare",
        "Teladoc",
    ],
    "Personal Care": [
        "Great Clips", "Sport Clips", "Supercuts", "Ulta Beauty", "Sephora",
        "Bath & Body Works", "Planet Fitness", "Crunch Fitness", "LA Fitness",
        "Orangetheory Fitness", "Gold's Gym", "Anytime Fitness", "Massage Envy",
        "European Wax Center", "Sally Beauty", "Dollar Shave Club", "Peloton",
        "ClassPass", "Lush", "Aveda", "Drybar", "Regis Salons", "Fantastic Sams",
    ],
    "Education": [
        "UF Bookstore", "UF Print", "University of Florida", "ChatGPT Plus", "OpenAI",
        "Perplexity", "Claude Pro", "Chegg", "Quizlet", "Coursera", "Udemy",
        "Grammarly", "Duolingo", "Pearson", "McGraw Hill", "Cengage", "Course Hero",
        "Barnes & Noble College", "Follett", "College Board", "ETS", "Kaplan",
        "Princeton Review", "LSAC", "AAMC",
    ],
    "Nightlife": [
        "The Locker Room", "MacDinton's", "Rush", "Grove", "Total Wine & More",
        "ABC Fine Wine & Spirits",
    ],
    "Other": [
        "Geico", "State Farm", "Progressive", "Allstate", "Verizon", "T-Mobile", "AT&T",
        "Mint Mobile", "Visible", "USPS", "The UPS Store", "FedEx", "iCloud",
        "Google One", "Dropbox", "GitHub", "Adobe Creative Cloud", "Patreon",
        "GoFundMe", "Red Cross", "Monthly Maintenance Fee", "Overdraft Fee",
        "Foreign Transaction Fee", "Interest Charge", "Late Fee", "Florida DHSMV",
        "Notary", "Dry Cleaning", "Laundromat", "NordVPN", "LinkedIn Premium", "Zoom",
        "Canva",
    ],
}

# Special spellings a real statement uses. Each extra is its own description.
EXTRA = {
    "Publix": ["PUBLIX SUPER MARKETS", "PUBLIX SUPERMARKET"],
    "Walmart": ["WAL-MART", "WAL-MART SUPERCENTER", "WALMART SUPERCENTER", "WALMART NEIGHBORHOOD MARKET"],
    "Whole Foods": ["WHOLE FOODS MARKET", "WHOLEFDS"],
    "Costco": ["COSTCO WHSE", "COSTCO WHOLESALE"],
    "Instacart": ["INSTACART*ORDER", "INSTACART HTTPSINSTACAR"],
    "Uber": ["UBER *TRIP", "UBER TRIP HELP.UBER.COM"],
    "Uber Eats": ["UBER *EATS", "UBER *EATS PENDING", "UBEREATS"],
    "Lyft": ["LYFT *RIDE", "LYFT RIDE"],
    "DoorDash": ["DOORDASH*ORDER", "DD *DOORDASH", "DOORDASH, INC."],
    "Grubhub": ["GRUBHUB*ORDER", "GRUBHUB"],
    "McDonald's": ["MCDONALDS", "MCDONALD'S F1234"],
    "Starbucks": ["STARBUCKS STORE", "STARBUCKS COFFEE"],
    "Chick-fil-A": ["CHICK-FIL-A", "CHICK FIL A", "CHICKFILA"],
    "Gator Dining": ["UF DINING", "UF HOUSING DINING"],
    "Amazon": ["AMAZON.COM", "AMZN Mktp US", "AMAZON PRIME", "AMAZON.COM*2K4AB9XY0"],
    "Netflix": ["NETFLIX.COM", "NETFLIX"],
    "Spotify": ["SPOTIFY USA", "SPOTIFY P1234567"],
    "Hulu": ["HULU", "HULU 877-8244858"],
    "Disney Plus": ["DISNEY PLUS", "DISNEYPLUS"],
    "Apple Music": ["APPLE MUSIC", "APPLE.COM/BILL MUSIC"],
    "Gainesville Regional Utilities": ["GRU", "GRU UTILITIES", "GAINESVILLE REGIONAL UTIL"],
    "FPL": ["FLORIDA POWER & LIGHT", "FPL DIRECT DEBIT"],
    "Water Bill": ["WATER BILL", "WATER & SEWER BILL"],
    "Electric Bill": ["ELECTRIC BILL", "ELECTRIC UTILITY BILL"],
    "UF Bookstore": ["UF BOOKSTORE", "UNIVERSITY OF FLORIDA BOOKSTORE", "UF BOOK STORE"],
    "UF Print": ["UF PRINT", "UF PRINTING"],
    "University of Florida": ["UNIV OF FLORIDA", "UNIVERSITY OF FLORIDA TUITION", "UF STUDENT ACCOUNT"],
    "ChatGPT Plus": ["CHATGPT PLUS", "OPENAI *CHATGPT SUBSCR", "OPENAI CHATGPT"],
    "OpenAI": ["OPENAI *CHATGPT", "OPENAI"],
    "CVS Pharmacy": ["CVS/PHARMACY", "CVS PHARMACY", "CVS"],
    "Walgreens": ["WALGREENS", "WALGREENS PHARMACY"],
    "Planet Fitness": ["PLANET FITNESS", "PLANET FIT", "PLANET FITNESS CLUB FEES"],
    "Sport Clips": ["SPORT CLIPS", "SPORTCLIPS", "SPORTS CLIPS"],
    "The Locker Room": ["THE LOCKER ROOM", "LOCKER ROOM"],
    "MacDinton's": ["MACDINTONS", "MACDINTON'S IRISH PUB"],
    "Rush": ["RUSH", "RUSH NIGHTCLUB"],
    "Grove": ["GROVE", "THE GROVE"],
    "Venmo": ["VENMO"],
    "Tropical Smoothie Cafe": ["TROPICAL SMOOTHIE CAFE", "TROPICAL SMOOTHIE"],
    "Dunkin": ["DUNKIN", "DUNKIN DONUTS"],
}

# Online services: no store numbers or city names on the statement.
ONLINE = {
    "Uber", "Uber Eats", "Lyft", "DoorDash", "Grubhub", "Instacart", "Amazon",
    "Shein", "Temu", "eBay", "Etsy", "Netflix", "Spotify", "Hulu", "Disney Plus",
    "HBO Max", "Peacock", "Paramount Plus", "YouTube Premium", "Apple Music",
    "Apple TV", "Amazon Prime Video", "Audible", "Steam", "PlayStation Network",
    "Xbox", "Nintendo", "Twitch", "Crunchyroll", "Epic Games", "Roblox",
    "Discord Nitro", "SiriusXM", "Pandora", "ChatGPT Plus", "OpenAI", "Perplexity",
    "Claude Pro", "Chegg", "Quizlet", "Coursera", "Udemy", "Grammarly", "Duolingo",
    "Course Hero", "Expedia", "Booking.com", "Airbnb", "iCloud", "Google One",
    "Dropbox", "GitHub", "Adobe Creative Cloud", "Patreon", "NordVPN",
    "LinkedIn Premium", "Zoom", "Canva", "Zocdoc", "Hims", "GoodRx", "Teladoc",
    "Dollar Shave Club", "Peloton", "ClassPass", "Water Bill", "Electric Bill",
    "Monthly Maintenance Fee", "Overdraft Fee", "Foreign Transaction Fee",
    "Interest Charge", "Late Fee", "Lemonade", "RentCafe", "AppFolio",
    "Zillow Rentals", "Apartments.com", "Ticketmaster", "StubHub", "SeatGeek",
    "Fandango", "Spectrum", "Xfinity", "Comcast", "AT&T Internet", "Verizon",
    "T-Mobile", "AT&T", "Mint Mobile", "Visible", "Geico", "State Farm",
    "Progressive", "Allstate", "Aetna", "Blue Cross Blue Shield", "Cigna",
    "UnitedHealthcare", "OpenAI", "Pearson", "McGraw Hill", "Cengage", "SunPass",
    "College Board", "ETS", "Kaplan", "Princeton Review", "LSAC", "AAMC",
    "Gainesville Regional Utilities", "Duke Energy", "FPL", "GoFundMe", "Red Cross",
    "Florida DHSMV", "University of Florida", "UF Print",
}

# Small local places often show up with a payment-processor prefix.
LOCAL = {"The Locker Room", "MacDinton's", "Rush", "Grove"}
PROCESSOR_OK = {"Dining", "Nightlife", "Personal Care", "Shopping", "Entertainment"}

# ---------------------------------------------------------------------------
# 2. Generic businesses: made-up names + a word that says what the business is.
#    This teaches the model that words like PIZZA, SALON or APTS matter.
# ---------------------------------------------------------------------------
STEMS = [
    "JOES", "MAMAS", "GOLDEN", "LUCKY", "SUNRISE", "OAK", "BLUE DOOR", "CORNER",
    "OLD TOWN", "GREEN LEAF", "RED BARN", "HARBOR", "ROYAL", "SILVER", "MIDNIGHT",
    "COPPER", "RUSTY", "IRON", "CRIMSON", "BLUE MOON", "DOWNTOWN", "SOUTHERN",
    "ISLAND", "SUNSET", "RIVERSIDE", "MAPLE", "PINE", "CAMPUS", "UNIVERSITY",
    "GATOR", "SWAMP", "TWIN OAKS", "LAKESIDE", "FIRST", "PRIME", "METRO",
]

GENERIC_TYPES = {
    "Dining": ["PIZZA", "GRILL", "CAFE", "COFFEE", "DINER", "KITCHEN", "TACOS", "SUSHI",
               "BBQ", "BAKERY", "DELI", "BURGERS", "WINGS", "BISTRO", "NOODLE HOUSE",
               "CREAMERY", "RESTAURANT", "SUBS"],
    "Groceries": ["MARKET", "SUPERMARKET", "GROCERY", "FOODS", "FRESH MARKET"],
    "Healthcare": ["DENTAL", "FAMILY DENTISTRY", "ORTHODONTICS", "PHARMACY", "URGENT CARE",
                   "MEDICAL CENTER", "PHYSICAL THERAPY", "DERMATOLOGY", "EYE CARE",
                   "CLINIC", "HEALTH CENTER"],
    "Personal Care": ["HAIR SALON", "NAIL SALON", "BARBERSHOP", "BARBER", "SPA", "WAXING",
                      "LASH STUDIO", "TANNING", "FITNESS", "GYM", "YOGA", "PILATES"],
    "Transportation": ["GAS", "FUEL", "AUTO REPAIR", "TIRE", "CAR WASH", "PARKING", "TAXI",
                       "AIRLINES", "AIRWAYS", "AUTO PARTS", "TOLL"],
    "Shopping": ["OUTFITTERS", "BOUTIQUE", "THRIFT", "SPORTING GOODS", "HARDWARE",
                 "ELECTRONICS", "FURNITURE", "SHOES", "CLOTHING"],
    "Entertainment": ["CINEMA", "THEATRE", "ARCADE", "BOWLING", "MINI GOLF", "ESCAPE ROOM",
                      "MUSEUM", "CONCERT TICKETS", "GAMES", "STREAMING"],
    "Education": ["TUTORING", "LEARNING CENTER", "ACADEMY", "TEST PREP", "BOOKSTORE",
                  "COLLEGE", "ONLINE COURSE"],
    "Nightlife": ["TAVERN", "PUB", "LOUNGE", "SALOON", "NIGHTCLUB", "BAR", "BREWING CO",
                  "HOOKAH LOUNGE", "COCKTAIL BAR", "SPORTS BAR", "LIQUOR"],
    "Housing": ["APTS", "APARTMENTS", "PROPERTIES", "PROPERTY MGMT", "RESIDENCES",
                "LIVING", "COMMONS", "LOFTS"],
    "Other": ["INSURANCE", "SHIPPING", "POSTAL", "DRY CLEANERS", "LAUNDRY", "NOTARY",
              "STORAGE", "FEE"],
}

# How many made-up names to create per category
GENERIC_COUNT = {
    "Dining": 70, "Groceries": 50, "Healthcare": 45, "Personal Care": 45,
    "Transportation": 35, "Shopping": 40, "Entertainment": 35, "Education": 30,
    "Nightlife": 90, "Housing": 20, "Other": 30,
}

CITIES = ["GAINESVILLE FL", "ORLANDO FL", "TAMPA FL", "JACKSONVILLE FL", "MIAMI FL",
          "TALLAHASSEE FL", "FT LAUDERDALE FL", "PLANTATION FL", "OCALA FL"]

STATEMENT_PREFIXES = ["POS PURCHASE ", "DEBIT CARD PURCHASE ", "PURCHASE ", "CHECKCARD "]
PROCESSOR_PREFIXES = ["SQ *", "TST* ", "PAYPAL *"]


def rand_id(low, high):
    n = rng.randint(low, high)
    return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def date_text():
    return f"{rng.randint(1, 12)}/{rng.randint(1, 28)}"


def make_variants(brand, how_many, online=False, local=False):
    """Return a set of statement-style descriptions for one brand."""
    up = brand.upper()
    forms = {up, brand, up.replace("'", ""), brand.title() if brand.islower() else brand}
    for extra in EXTRA.get(brand, []):
        forms.add(extra)
        forms.add(extra.title())

    # Always add the same number of noisy variants, no matter how many
    # special spellings a brand already has (otherwise brands with many
    # special spellings end up with too few rows).
    target = len(forms) + max(how_many - 4, 0)
    attempts = 0
    while len(forms) < target and attempts < 300:
        attempts += 1
        base = rng.choice([up, up, up.replace("'", ""), brand])
        if online:
            style = rng.choice(["ref", "prefix", "com", "prefix"])
            if style == "ref":
                text = f"{base}*{rand_id(6, 9)}"
            elif style == "com":
                text = f"{base.replace(' ', '')}.COM"
            else:
                text = rng.choice(STATEMENT_PREFIXES) + base
        else:
            style = rng.choice(["store", "store", "city", "both", "prefix", "prefix_city"])
            number = rng.randint(100, 9999)
            if local and rng.random() < 0.5:
                style = "processor"
            if style == "store":
                text = rng.choice([f"{base} #{number}", f"{base} {number}", f"{base} STORE {number}"])
            elif style == "city":
                text = f"{base} {rng.choice(CITIES)}"
            elif style == "both":
                text = f"{base} #{number} {rng.choice(CITIES)}"
            elif style == "prefix":
                text = rng.choice(STATEMENT_PREFIXES) + base
            elif style == "processor":
                text = rng.choice(PROCESSOR_PREFIXES) + base
            else:
                text = rng.choice(STATEMENT_PREFIXES) + f"{base} {rng.choice(CITIES)}"
        if rng.random() < 0.08:
            text = f"{text} {date_text()}"
        forms.add(text)
    return forms


# ---------------------------------------------------------------------------
# 3. Rent: junk letters and numbers first, the word RENT last
# ---------------------------------------------------------------------------
RENT_PROPERTIES = [
    "CAMPUS VIEW APTS", "COLLEGE PARK LIVING", "OAK HOLLOW RESIDENCES", "MAPLE COMMONS",
    "UNIVERSITY LOFTS", "THE COMMONS", "STONE CREEK PROPERTIES", "PINE RIDGE APARTMENTS",
    "SUNSET LIVING", "RIVER OAKS", "LAKESIDE APTS", "CAMPUS CROSSING",
]
RENT_PREFIXES = ["WEB PMT", "ACH PMT", "ONLINE PMT", "AUTOPAY", "PMT", "ELECTRONIC PMT", "WEB ID"]
PLAIN_RENT = ["RENT", "Rent", "APARTMENT RENT", "Apartment Rent", "RENT PAYMENT",
              "Rent Payment", "MONTHLY RENT", "ROOMMATE RENT"]


def rent_rows(how_many):
    rows = []
    for text in PLAIN_RENT:
        rows.append((text, "Housing", "Rent"))
    seen = set(PLAIN_RENT)
    while len(rows) < how_many:
        prefix = rng.choice(RENT_PREFIXES)
        prop = rng.choice(RENT_PROPERTIES)
        code = rand_id(5, 10)
        style = rng.choice(["a", "b", "c", "d", "e", "f"])
        if style == "a":
            text = f"{prefix} {code} RENT"
        elif style == "b":
            text = f"{prefix} {code} {prop} RENT"
        elif style == "c":
            text = f"{code} {prop} RENT"
        elif style == "d":
            text = f"{prop} RENT"
        elif style == "e":
            text = f"{prefix} {prop} {code} RENT"
        else:
            text = f"{prefix} {code} Rent"
        if text not in seen:
            seen.add(text)
            rows.append((text, "Housing", prop if style != "a" else "Rent"))
    return rows


# ---------------------------------------------------------------------------
# 4. Transfers: Venmo, Zelle, Cash App, ATM, bank transfers
# ---------------------------------------------------------------------------
NAMES = ["MIKE S", "SARAH K", "JAMES L", "EMILY R", "DAVID T", "JESSICA M", "CHRIS P",
         "ASHLEY W", "JOHN D", "MARIA G", "ALEX B", "TYLER H", "OLIVIA C", "NOAH F",
         "SOPHIA N", "ETHAN J", "ISABELLA V", "LIAM Q", "MIA Z", "DANIEL A", "GRACE E",
         "ANDREW Y", "NATALIE O", "KEVIN U", "LAUREN I", "BRANDON X"]


def transfer_rows():
    rows = []

    def add(text, brand):
        rows.append((text, "Transfers", brand))

    for text in ["VENMO PAYMENT", "VENMO CASHOUT", "VENMO", "Venmo", "Venmo Payment",
                 "Venmo Cashout"]:
        add(text, "Venmo")
    for name in NAMES:
        add(f"VENMO *{name}", "Venmo")
        add(f"VENMO PAYMENT {rand_id(8, 10)}", "Venmo")
        add(f"ZELLE PAYMENT TO {name}", "Zelle")
        add(f"ZELLE PAYMENT FROM {name}", "Zelle")
        add(f"ZELLE TO {name}", "Zelle")
        add(f"Zelle to {name.title()}", "Zelle")
        add(f"CASH APP*{name}", "Cash App")
        add(f"CASH APP*{rand_id(6, 8)}", "Cash App")
        add(f"PAYPAL *{name}", "PayPal")
        add(f"APPLE CASH SENT MONEY {name}", "Apple Cash")
    for text in ["ZELLE", "Zelle", "CASH APP", "Cash App", "CASH APP*CASH OUT", "PAYPAL TRANSFER",
                 "PAYPAL INST XFER", "PAYPAL", "APPLE CASH", "APPLE CASH SENT MONEY",
                 "APPLE CASH CARD ADD"]:
        brand = "Cash App" if "CASH APP" in text.upper() else "PayPal" if "PAYPAL" in text.upper() \
            else "Apple Cash" if "APPLE" in text.upper() else "Zelle"
        add(text, brand)
    for text in ["ONLINE TRANSFER TO SAVINGS", "ONLINE TRANSFER FROM CHECKING",
                 "TRANSFER TO SAVINGS", "TRANSFER FROM SAVINGS", "ACH TRANSFER",
                 "ONLINE BANKING TRANSFER", "Online Transfer", "Transfer to Savings",
                 "MOBILE TRANSFER", "INTERNAL TRANSFER"]:
        add(text, "Bank Transfer")
    for city in CITIES:
        add(f"ATM WITHDRAWAL {city}", "ATM")
        add(f"ATM CASH WITHDRAWAL {city}", "ATM")
    for text in ["ATM WITHDRAWAL", "ATM CASH WITHDRAWAL", "ATM Withdrawal", "CASH WITHDRAWAL",
                 "NON-BANK ATM WITHDRAWAL"]:
        add(text, "ATM")
    return rows


# ---------------------------------------------------------------------------
# 5. Build the file
# ---------------------------------------------------------------------------
def main():
    rows = []

    for category, brands in BRANDS.items():
        for brand in brands:
            how_many = 12
            online = brand in ONLINE
            local = brand in LOCAL
            for text in make_variants(brand, how_many, online=online, local=local):
                rows.append((text, category, brand))

    for category, count in GENERIC_COUNT.items():
        types = GENERIC_TYPES[category]
        used = set()
        while len(used) < count:
            used.add(f"{rng.choice(STEMS)} {rng.choice(types)}")
        for name in sorted(used):
            for text in make_variants(name, 8, local=(category in PROCESSOR_OK)):
                rows.append((text, category, name))

    rows += rent_rows(160)
    rows += transfer_rows()

    # One description must have one category. Drop repeats and any conflicts.
    first_category = {}
    conflicts = set()
    for text, category, brand in rows:
        key = text.strip().upper()
        if key in first_category and first_category[key] != category:
            conflicts.add(key)
        first_category.setdefault(key, category)

    clean, seen = [], set()
    for text, category, brand in rows:
        key = text.strip().upper()
        if key in conflicts:
            continue
        if (text, category) in seen:
            continue
        seen.add((text, category))
        clean.append((text.strip(), category, brand))

    rng.shuffle(clean)
    with open("data/real_merchants.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["description", "category", "brand"])
        writer.writerows(clean)

    counts = {}
    for _, category, _ in clean:
        counts[category] = counts.get(category, 0) + 1
    print(f"Wrote {len(clean)} rows to data/real_merchants.csv")
    for category, n in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {category}: {n}")
    if conflicts:
        print("Dropped descriptions that matched two categories:", len(conflicts))


if __name__ == "__main__":
    main()
