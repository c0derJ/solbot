"""
SOLBOT - News Scraper & Sentiment Engine
Scrapes @solidintel_x and scores sentiment for SOL/USD
"""

import os
import re
import time
import logging
from datetime import datetime

log = logging.getLogger(__name__)

# SOL-specific keyword weights
BULL_KEYWORDS = {
    'bullish': 2, 'breakout': 2, 'pumping': 1.5, 'surge': 1.5,
    'rally': 1.5, 'moon': 1, 'buy': 1, 'long': 1, 'support': 1,
    'accumulate': 1.5, 'bounce': 1, 'green': 0.5, 'ath': 2,
    'upgrade': 1.5, 'launch': 1, 'adoption': 1.5, 'partnership': 1.5,
    'recover': 1, 'higher': 0.5, 'bottom': 1, 'reversal': 1,
    'defi': 0.5, 'nft': 0.5, 'ecosystem': 0.5, 'tvl': 1,
}

BEAR_KEYWORDS = {
    'bearish': 2, 'dump': 2, 'crash': 2, 'sell': 1, 'short': 1,
    'resistance': 1, 'rejection': 1.5, 'breakdown': 2, 'liquidation': 2,
    'red': 0.5, 'concern': 1, 'warning': 1.5, 'risk': 1,
    'hack': 2.5, 'exploit': 2.5, 'scam': 2, 'fud': 1.5,
    'falling': 1, 'lower': 0.5, 'drop': 1.5, 'plunge': 2,
    'whale': 0.5, 'outflow': 1.5, 'exchange': 0.5,
}

SOL_KEYWORDS = ['sol', 'solana', '$sol', 'solana network', 'phantom', 'raydium', 'jito']


class NewsScraper:
    def __init__(self, account='solidintel_x'):
        self.account = account
        self.last_tweets = []
        self.last_score = 0.0
        self.last_update = None
        self.scraper = None
        self._init_scraper()

    def _init_scraper(self):
        """Initialize ntscraper with fallback."""
        try:
            from ntscraper import Nitter
            self.scraper = Nitter(log_level=0)
            log.info(f"ntscraper initialized for @{self.account}")
        except ImportError:
            log.warning("ntscraper not installed — using mock data")
        except Exception as e:
            log.warning(f"ntscraper init failed: {e} — using mock data")

    def fetch_tweets(self, limit=10):
        """Fetch latest tweets from the account."""
        if self.scraper is None:
            return self._mock_tweets()

        try:
            result = self.scraper.get_tweets(self.account, mode='user', number=limit)
            tweets = []
            if result and 'tweets' in result:
                for t in result['tweets'][:limit]:
                    text = t.get('text', '')
                    tweets.append({
                        'text': text,
                        'time': t.get('date', datetime.now().isoformat()),
                        'link': t.get('link', ''),
                    })
            log.info(f"Fetched {len(tweets)} tweets from @{self.account}")
            return tweets if tweets else self._mock_tweets()
        except Exception as e:
            log.warning(f"Scrape failed: {e} — using mock data")
            return self._mock_tweets()

    def _mock_tweets(self):
        """Fallback mock tweets for paper trading / testing."""
        return [
            {'text': 'SOL looking strong at key support. Accumulate here.', 'time': datetime.now().isoformat(), 'link': ''},
            {'text': 'Watch $SOL — breakout incoming if BTC holds $60k', 'time': datetime.now().isoformat(), 'link': ''},
            {'text': 'Solana ecosystem TVL hitting new highs. Bullish on SOL mid-term.', 'time': datetime.now().isoformat(), 'link': ''},
        ]

    def score_tweet(self, text):
        """Score a single tweet: positive = bullish, negative = bearish."""
        text_lower = text.lower()
        bull = 0.0
        bear = 0.0

        # Check SOL relevance
        sol_relevant = any(kw in text_lower for kw in SOL_KEYWORDS)
        multiplier = 1.5 if sol_relevant else 1.0

        for kw, weight in BULL_KEYWORDS.items():
            if kw in text_lower:
                bull += weight

        for kw, weight in BEAR_KEYWORDS.items():
            if kw in text_lower:
                bear += weight

        # Emoji boosts
        bull += text.count('🚀') * 1 + text.count('🟢') * 0.5 + text.count('📈') * 1
        bear += text.count('🔴') * 0.5 + text.count('📉') * 1 + text.count('⚠️') * 1

        raw = (bull - bear) * multiplier
        # Normalize to -1 to +1
        max_val = max(abs(raw), 1)
        score = max(-1.0, min(1.0, raw / max_val))
        return round(score, 3)

    def analyze(self):
        """Fetch tweets, score sentiment, return analysis dict."""
        tweets = self.fetch_tweets(limit=10)
        self.last_tweets = tweets

        if not tweets:
            self.last_score = 0.0
            return self._build_result([], 0.0)

        scores = [self.score_tweet(t['text']) for t in tweets]
        avg_score = sum(scores) / len(scores)
        self.last_score = round(avg_score, 3)
        self.last_update = datetime.now().isoformat()

        enriched = []
        for tweet, score in zip(tweets, scores):
            sentiment = 'bull' if score > 0.1 else 'bear' if score < -0.1 else 'neutral'
            impact = 'high' if abs(score) > 0.6 else 'medium' if abs(score) > 0.3 else 'low'
            enriched.append({
                **tweet,
                'score':     score,
                'sentiment': sentiment,
                'impact':    impact,
            })

        log.info(f"Sentiment score: {self.last_score:+.3f} from {len(tweets)} tweets")
        return self._build_result(enriched, self.last_score)

    def _build_result(self, tweets, score):
        signal = 'BULLISH' if score > 0.2 else 'BEARISH' if score < -0.2 else 'NEUTRAL'
        return {
            'score':       score,
            'signal':      signal,
            'tweets':      tweets,
            'tweet_count': len(tweets),
            'account':     self.account,
            'updated':     self.last_update or datetime.now().isoformat(),
        }
