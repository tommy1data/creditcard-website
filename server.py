#!/usr/bin/env python3
"""
CardCompare AI Card Finder — Backend Server (Gemini)
=====================================================
Requires: pip install fastapi uvicorn google-generativeai

Set your Gemini API key as environment variable:
  export GEMINI_API_KEY=AIza...

Then run:
  python server.py
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini client — uses GEMINI_API_KEY env variable
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

# ====== CHANGE MODEL NAME HERE IF NEEDED ======
# Options: gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro
# See https://ai.google.dev/gemini-api/docs/models
MODEL_NAME = "gemini-2.5-flash"

CARD_DATA = """
ALL CARDS DATABASE (as of March 2026):

=== CHASE ===
1. Chase Debit (Total Checking): $0-$15/mo fee (waivable w/ $500 direct deposit), 3% foreign tx fee, 14,000+ ATMs, no rewards, no credit check needed.
2. Chase Freedom Flex: $0 fee, 5% rotating quarterly categories (up to $1,500/qtr), 5% Chase Travel, 3% dining/drugstores, 1% all else, $200 bonus/$500 spend, 0% intro APR 15mo, cell phone protection $800, 3% foreign tx, credit 690+.
3. Chase Freedom Unlimited: $0 fee, 1.5% flat all purchases, 5% Chase Travel, 3% dining/drugstores, $250 bonus/$500 spend, 0% intro APR 15mo, 3% foreign tx, credit 690+.
4. Chase Sapphire Preferred: $95 fee, 5x Chase Travel, 3x dining/streaming/online grocery, 2x travel, 1x all else, 75,000 pts bonus/$5K spend, primary rental car ($60K), $10K trip cancel, no foreign tx, credit 690+.
5. Chase Sapphire Reserve: $795 fee, 8x Chase Travel, 4x direct flights/hotels, 3x dining, 1x else, 125,000 pts bonus/$6K spend, ~$2,718 in annual credits, Priority Pass lounges, primary rental ($75K), $1M travel accident, no foreign tx, credit 720+.

=== BANK OF AMERICA ===
1. BofA Debit (Advantage Banking): $4.95-$25/mo (waivable), 3% foreign tx, BankAmeriDeals cash back, no credit check.
2. BofA Customized Cash Rewards: $0 fee, 3% choice category (6% yr1), 2% grocery/wholesale, 1% all else, $200 bonus/$1K, 0% intro 15mo, 3% foreign tx, credit 670+. Up to 5.25% w/ Preferred Rewards.
3. BofA Unlimited Cash Rewards: $0 fee, 1.5% flat (2% yr1), $200 bonus/$1K, 0% intro 15mo, 3% foreign tx, credit 670+. Up to 2.625% w/ Preferred Rewards.
4. BofA Travel Rewards: $0 fee, 1.5x all (3x BofA Travel Center), 25K pts bonus/$1K, 0% intro 15mo, no foreign tx, credit 670+.
5. BofA Premium Rewards: $95 fee, 2x travel/dining, 1.5x all else, 60K pts bonus/$4K, $100 airline credit, no foreign tx, credit 670+.
6. BofA Premium Rewards Elite: $550 fee, 2x travel/dining, 1.5x else, 75K pts bonus/$5K, $300 airline + $150 lifestyle credits, Priority Pass, no foreign tx.

=== AMERICAN EXPRESS ===
1. Blue Cash Everyday: $0 fee, 3% grocery (up to $6K/yr), 3% gas, 3% online retail, 1% else, $200 bonus/$2K, 0% intro 15mo, 2.7% foreign tx, credit 670+.
2. Blue Cash Preferred: $95 fee, 6% grocery (up to $6K/yr), 6% streaming, 3% transit/gas, 1% else, $350 bonus/$3K, 0% intro 12mo, 2.7% foreign tx, credit 670+.
3. Amex Green: $150 fee, 3x travel/transit/dining, 1x else, 40K pts bonus/$3K, $200 CLEAR Plus credit, no foreign tx, MR transfer partners, credit 690+.
4. Amex Gold: $325 fee, 4x restaurants/grocery (up to $25K/yr), 3x flights, 1x else, 60K pts bonus/$6K, $120 Uber + $120 dining + $120 Dunkin credits, no foreign tx, credit 700+.
5. Amex Platinum: $895 fee, 5x flights (direct + Amex Travel), 5x prepaid hotels, 1x else, up to 175K pts bonus/$12K/6mo, $200 airline + $200 hotel + $240 entertainment + $200 Uber + more credits, Centurion Lounges, Priority Pass, no foreign tx, credit 720+.

=== CAPITAL ONE ===
1. Capital One 360 Checking (Debit): $0 fee, 70K+ ATMs, 0% foreign tx, ITIN friendly, 0.10% APY, no credit check.
2. Capital One Platinum Secured: $0 fee, $49-$200 deposit for $200 credit line, no rewards, graduation path, ITIN accepted, no credit needed.
3. Capital One Quicksilver Secured: $0 fee, $200 deposit, 1.5% cash back, $50 bonus, graduation path, ITIN accepted, limited/no credit.
4. Capital One Quicksilver: $0 fee, 1.5% flat cash back, 5% Cap1 Travel, $200 bonus/$500 spend, 0% intro 15mo, no foreign tx, credit 690+.
5. Capital One SavorOne: $0 fee, 3% dining/grocery/streaming/entertainment, 8% Cap1 Entertainment, 5% Cap1 Travel, 1% else, $200 bonus/$500, no foreign tx, credit 690+.
6. Capital One Venture: $95 fee, 2x all purchases, 5x Cap1 Travel, 75K miles + $250 hotel credit bonus/$4K, TSA/GE $120 credit, transfer to 22 partners, no foreign tx, credit 700+.
7. Capital One Venture X: $395 fee, 2x all, 5x hotels/rental via Cap1 Travel, 10x Cap1 Travel, 75K miles bonus/$4K, $300 travel credit + 10K anniversary miles, Capital One Lounges, Priority Pass, no foreign tx, credit 740+.

=== WELLS FARGO ===
1. Wells Fargo Debit (Everyday Checking): $10/mo (waivable w/ $500 DD or $1,500 balance), ~10K ATMs, 3% foreign tx, no credit check.
2. Wells Fargo Active Cash: $0 fee, 2% flat cash back, $200 bonus/$500, 0% intro 12mo, $600 cell phone protection, 3% foreign tx, credit 690+.
3. Wells Fargo Reflect: $0 fee, 0% APR for 21 months (purchase+BT), no rewards, $600 cell phone protection, 3% foreign tx, credit 690+. Best balance transfer card.
4. Wells Fargo Autograph: $0 fee, 3x dining/travel/gas/transit/streaming/phone plans, 1x else, 20K pts bonus/$1K/3mo, no foreign tx, transfer partners, $600 cell phone protection, credit 690+.
5. Wells Fargo Autograph Journey: $95 fee, 5x hotels, 4x airlines, 3x dining/other travel, 1x else, 60K pts bonus/$4K, $50 airline credit, $1,000 cell phone protection, $1M travel accident, no foreign tx, credit 720+.

=== CITI ===
1. Citi Debit (Basic Checking): $12/mo (waivable), Citi ATMs free, 3% foreign tx, no credit check.
2. Citi Secured Mastercard: $0 fee, $200-$2,500 deposit, no rewards, 26.74% APR, graduation after 18mo, ITIN accepted, no credit needed.
3. Citi Double Cash: $0 fee, 2% flat (1% on purchase + 1% on payment), 0% BT intro 18mo, 3% foreign tx, credit 690+. No sign-up bonus.
4. Citi Custom Cash: $0 fee, 5% on top spending category auto-detected (up to $500/billing cycle), 1% else, $200 bonus/$1.5K, 0% intro 15mo, 3% foreign tx, credit 690+.
5. Citi Strata Premier: $95 fee, 3x air/hotels/restaurants/supermarkets/gas/EV, 1x else, 75K pts bonus/$4K, $120 hotel credit/yr, no foreign tx, transfer to 19 partners, credit 700+.

=== CHARLES SCHWAB ===
1. Schwab Investor Checking (Debit): $0 fee, unlimited worldwide ATM rebates, 0% foreign tx, linked to brokerage, FDIC insured, 0.01% APY. Best debit card for international travel.
2. Schwab Roth IRA: $0 fees, $7,500 contribution limit (2026), tax-free growth, no RMDs, income limits apply.
3. Schwab Traditional IRA: $0 fees, $7,500 limit, tax-deductible contributions, RMDs at 73.
4. Schwab Brokerage: $0 commissions stocks/ETFs, fractional shares, $0 minimum, SIPC + $150M excess.
5. Schwab Intelligent Portfolios: $0 advisory fee ($5K min) or $300+$30/mo Premium ($25K min).

=== SECURED CARDS (for building credit / no SSN) ===
- Capital One Platinum Secured: $0 fee, $49-$200 deposit, no rewards, ITIN accepted
- Capital One Quicksilver Secured: $0 fee, $200 deposit, 1.5% cash back, ITIN accepted
- Citi Secured Mastercard: $0 fee, $200-$2,500 deposit, no rewards, ITIN accepted
- Wells Fargo: NO secured card available (discontinued 2019)
- Chase: NO secured card available
- BofA: NO secured card available for general public
- Amex: NO traditional secured card (but some pre-qualified offers exist)

=== CARDS AVAILABLE WITH ITIN (no SSN needed) ===
- Capital One: 360 Checking, Platinum Secured, Quicksilver Secured, and most other cards
- Citi: Secured Mastercard, and potentially other cards
- Bank of America: Some cards accept ITIN
- Wells Fargo: Accepts ITIN but harder for non-permanent residents
- Chase: Generally requires SSN
- Amex: Generally requires SSN

=== BEST SIGN-UP BONUSES (by value) ===
1. Amex Platinum: up to 175,000 MR pts (~$3,500+ value)
2. Chase Sapphire Reserve: 125,000 UR pts (~$2,500+ value)
3. Citi Strata Premier: 75,000 TY pts (~$1,200+ value)
4. Capital One Venture: 75,000 miles + $250 credit (~$1,300+ value)
5. Chase Sapphire Preferred: 75,000 UR pts (~$937+ value)
"""

SYSTEM_PROMPT = f"""You are a credit card advisor embedded in the CardCompare website. You help users find the best cards based on their profile.

You have access to this card database:
{CARD_DATA}

RULES:
1. Always recommend cards from the database above — never invent cards.
2. Consider the user's credit score, spending habits, goals, existing cards, and citizenship/residency status.
3. For users without SSN or with limited credit history, prioritize secured cards and ITIN-friendly issuers (Capital One, Citi).
4. For students, suggest no-annual-fee starter cards first.
5. For frequent travelers, consider foreign transaction fees, lounge access, and travel protections.
6. Rank recommendations from best to acceptable. Explain WHY each card fits.
7. If credit score is below 670, focus on secured cards and credit-building options.
8. If credit score is 670-720, suggest mid-tier cards (no-fee rewards cards).
9. If credit score is 720+, include premium options.
10. Mention the "card ladder" — how to progress from starter to premium cards over time.
11. Always mention the sign-up bonus value when relevant.
12. Be concise but thorough. Use markdown formatting for readability.
13. Answer in the same language the user writes in (English or Russian).
14. Never provide financial advice — only information. Remind users to verify terms on official bank websites.
"""


@app.post("/api/recommend")
async def recommend(request: Request):
    try:
        body = await request.json()
        user_msg = body.get("message", "")
        history = body.get("history", [])

        # Build Gemini chat history (role must be "user" or "model")
        contents = []
        for h in history[-10:]:
            role = "model" if h["role"] == "assistant" else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=h["content"])])
            )
        contents.append(
            types.Content(role="user", parts=[types.Part(text=user_msg)])
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=2048,
            ),
            contents=contents,
        )
        reply = response.text
        return JSONResponse({"reply": reply})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"reply": f"Error: {str(e)}"}, status_code=500)


@app.get("/api/models")
async def list_models():
    try:
        models = [m.name for m in client.models.list() if "generateContent" in (m.supported_actions or [])]
        return {"models": models}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
