# Stock Selection Strategy for Training Bottom Detection Model

## The Core Problem We're Solving

When you want to train a model to predict stock bottoms, your instinct might be: "Just grab all S&P 500 stocks and download 10 years of data." This seems logical - you get large, well-known companies with reliable data.

**But this creates a fatal flaw: Survivorship Bias.**

The S&P 500 TODAY contains only companies that SURVIVED the last 10 years. You're training on winners. When your model encounters a struggling company in real trading, it will see patterns it's never encountered during training and fail catastrophically. It learned that "every bottom eventually recovers" because it never saw companies that went bankrupt.

---

## The Strategy: Point-in-Time Universe with Failure Inclusion

### Phase 1: Start With Known Quality (S&P 500)

**What:** Use all 503 stocks in your S&P 500 list as the foundation.

**Why:** These are proven, liquid, large-cap stocks with reliable data. They form a quality baseline. The S&P 500 historically has been rebalanced - companies get added and removed. While your list represents current constituents, these are the highest-quality stocks available.

**How it helps:** Gives you a solid core of 500 well-behaved, liquid stocks that definitely have good data. These will form the "normal" patterns in your training set.

---

### Phase 2: Sample Broader Universe Intelligently

**What:** From your 11,917 US tickers, filter out junk (warrants, units, preferred shares, ETFs), leaving ~9,000 clean tickers. Then sample ~2,000 of these, prioritizing shorter ticker symbols.

**Why shorter symbols?** Companies listed on major exchanges (NYSE, NASDAQ) typically have 1-5 letter tickers. Over-the-counter (OTC) stocks and junk companies often have longer tickers or special characters. By prioritizing shorter symbols, you naturally filter toward exchange-listed companies without needing to know which exchange each trades on.

**Why sample instead of downloading all 9,000?** Downloading 9,000 stocks takes hours and many will be rejected anyway (illiquid, penny stocks). Sampling 2,000 gives you diversity while keeping download time reasonable (~45 minutes). You'll still get 1,500+ good stocks after quality filtering.

**How it helps:** Expands beyond just large-caps to include mid-caps and small-caps, giving your model exposure to different market dynamics. A $2 billion company behaves differently from a $200 billion company.

---

### Phase 3: Quality Filtering (The Gatekeeper)

**What:** For each downloaded stock, check:
- Average price > $1 (reject penny stocks)
- Average volume > 50,000 shares/day (reject illiquid)
- At least 1 year of data (reject insufficient history)

**Why reject penny stocks?** Stocks under $1 have unreliable technical patterns. They're manipulated easily, have wide bid-ask spreads, and their "bottoms" don't follow normal market dynamics. Training on these teaches your model nonsense patterns.

**Why reject illiquid stocks?** If you can't trade 100,000 shares without moving the price significantly, the stock is useless for practical trading. Volume under 50K/day means you can't actually execute trades at the "bottoms" your model predicts. You'll identify the bottom but can't profit from it.

**Why need 1+ years?** Technical indicators like 200-day moving averages need history. Also, with less than 252 trading days (1 year), you don't have enough data points to identify meaningful patterns. The stock might have just IPO'd or been delisted soon after listing.

**How it helps:** Ensures every stock in your training set represents a TRADEABLE opportunity. You're not learning from theoretical patterns that can't be executed.

---

### Phase 4: The Critical Innovation - Capture Failed Stocks

**What:** Identify stocks that stopped trading more than 90 days ago, or whose price collapsed from healthy levels (>$5 average) to penny stock territory (<$1).

**Why this is GOLD:** These are bankrupt, delisted, or acquired-in-distress companies. Most people's training sets have ZERO of these. But in real trading, you'll encounter struggling companies. If your model only learned from survivors, it thinks every bottom is a buying opportunity. These failed stocks teach the model: "Some bottoms are death spirals, not opportunities."

**How it helps:** Eliminates survivorship bias. Your model learns two types of bottoms:
1. Recoverable bottoms (the survivors)
2. Point-of-no-return bottoms (the failures)

This is the difference between a toy model and a production model.

---

### Phase 5: Stratified Selection (The Balancing Act)

**What:** From your ~1,800 valid stocks and ~150 failed stocks, select 1,500 using this formula:

**Market Cap Allocation:**
- 15% mega-cap ($200B+): ~225 stocks
- 25% large-cap ($10B-200B): ~375 stocks  
- 30% mid-cap ($2B-10B): ~450 stocks
- 20% small-cap ($300M-2B): ~300 stocks
- 10% failed (any size): ~150 stocks

**Within each market cap bucket, balance by sector** (no sector >25% of bucket).

**Why stratify by market cap?** Different dynamics:
- Mega-caps ($200B+): Slow-moving, less volatile, institutional-dominated. Bottoms are gradual.
- Large-caps ($10B-200B): Moderate volatility, still institutional but more reactive.
- Mid-caps ($2B-10B): Higher growth potential, more volatile, bottoms can be sharp.
- Small-caps ($300M-2B): Very volatile, retail-driven, bottoms can be panic-driven.

If you train on 80% large-caps, your model fails on small-caps because it never learned their behavior patterns.

**Why balance sectors?** Tech stocks during a tech bubble behave differently than utilities. If your training set is 40% tech (because tech dominates S&P 500), your model learns tech patterns and fails on energy or healthcare stocks. Bottoms in defensive sectors (utilities, consumer staples) look different from bottoms in growth sectors (tech, biotech).

**Why include 10% failures across all sizes?** Because companies of ALL sizes fail. You need your model to recognize failure patterns in both $100B companies (rare but catastrophic) and $500M companies (more common).

**How it helps:** Your model sees the full spectrum of market behavior. It learns that:
- A 30% drop in a mega-cap is extreme (potential bottom)
- A 30% drop in a small-cap might be Tuesday (not necessarily a bottom)
- Tech bottoms have different volume patterns than bank bottoms
- Some bottoms recover (survivors), some don't (failures)

---

### Phase 6: Metadata Enrichment

**What:** For each selected stock, fetch current market cap, sector, and industry from yfinance.

**Why do this AFTER initial filtering?** Fetching metadata is slow (API rate limits). By filtering first (2,500 → 1,800), you only fetch metadata for stocks that passed quality checks, saving 10+ minutes.

**Why do we need this at all?** Two reasons:
1. **Stratification**: Can't allocate by market cap without knowing market caps
2. **Future feature engineering**: You might want to add "sector-relative performance" features later (how is this stock doing vs its sector?)

**How it helps:** Enables intelligent selection rather than random sampling. Also gives you metadata you can use as features if needed (though market cap and sector shouldn't be direct features - they're for understanding your data distribution).

---

## The Final Dataset: What You Get and Why It Matters

**Composition:**
- 1,500 stocks total
- Mix of all market caps (mega to small)
- Balanced sectors (no sector dominates)
- 10% failed companies (anti-survivorship bias)
- 2015-2024 data (10 years, ~2,500 trading days)
- ~2.7 million rows total (1,500 stocks × 1,800 days average)

**Why this specific composition?**

**The 1,500 number:** Large enough for diversity, small enough to manage. With 1:250 imbalance (bottoms are rare), you'll have:
- ~10,800 positive samples (bottoms) across all stocks
- ~2.7 million negative samples (non-bottoms)
- After SMOTE/resampling: enough data to train without overfitting

**The 10-year window:** Long enough to capture:
- Different market regimes (bull markets, bear markets, sideways)
- Different interest rate environments
- Multiple recession cycles
- Various sector rotations
- Tech boom and bust cycles

Too short (5 years) and you might only see bull market bottoms. Too long (20 years) and old data becomes less relevant to current market structure.

**The failure inclusion:** This 10% is worth its weight in gold. Consider:
- 1,350 survivors teach: "What do recoverable bottoms look like?"
- 150 failures teach: "What do death spirals look like?"

Without the failures, your model is like a doctor who only studied healthy patients. They can't recognize disease.

---

## Why NOT Just Use Random Selection?

**Scenario A: Random selection from your 11,917 tickers**

You'd get:
- 70% small-caps (most tickers are small companies)
- 40% tech (tech has most publicly traded companies)
- 50% illiquid (many tickers trade <10K shares/day)
- 20% penny stocks (many tickers are under $5)
- 0% failures (current listings only)

Your model would:
- Overfit to small-cap patterns
- Fail on large-caps
- Learn from untradeable stocks
- Never see bankruptcy patterns

**Scenario B: Stratified selection (our approach)**

You get:
- Balanced market caps (realistic trading universe)
- Balanced sectors (all market conditions)
- Only tradeable stocks (>50K volume)
- Only quality data (>$1 price)
- Includes failures (anti-survivorship)

Your model:
- Generalizes across market caps
- Works in all sectors
- Learns from tradeable opportunities
- Recognizes both recoveries and failures

---

## The Information Asymmetry Advantage

**What most people do:** Download current S&P 500, train model, get 85% accuracy in backtest.

**What they miss:** They're testing on survivors. They don't know their model fails on struggling companies because struggling companies aren't in their dataset.

**What you're doing:** Including failures explicitly. Your backtest accuracy might be lower (maybe 75%) because your model sees hard cases, but your REAL-WORLD performance will be better because the model learned to avoid death traps.

**Analogy:** It's like training a self-driving car:
- **Bad approach:** Train only on sunny days with perfect conditions. Get 99% accuracy in sunny weather, but crash in rain because it never learned rain exists.
- **Good approach:** Train on sunny days, rainy days, snow, fog, and night driving. Get 85% accuracy in testing, but handle ALL conditions in real world.

Including failed stocks is your "training in bad weather." It makes your backtest look worse but makes your real trading better.

---

## Summary: The Strategy in One Paragraph

Start with proven quality (S&P 500), expand to broader universe (sampled US stocks), filter for tradeability (liquidity, price, history), CRITICALLY include failures (delisted/bankrupt stocks), then stratify by market cap and sector to ensure balanced representation. The result is a training set that mirrors real-world trading: diverse market caps, all sectors, actually tradeable stocks, and crucially, examples of both recoverable bottoms and death spirals. This eliminates survivorship bias while maintaining data quality, giving your model the best chance to generalize to real trading conditions.

---

**The key insight:** A good training dataset isn't about maximizing quantity or even maximizing quality - it's about maximizing REPRESENTATIVENESS of what you'll encounter in real trading. Quality matters (tradeable stocks), but DIVERSITY and FAILURE INCLUSION matter more for building a robust model that works in production.