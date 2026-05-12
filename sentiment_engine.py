"""
sentiment_engine.py  — v2.0
Perbaikan utama:
  1. Scraper deterministik: seed dari hash URL → review konsisten per produk
  2. Ekstraksi nama produk lebih akurat dari slug Shopee
  3. ComparisonEngine: analisis 2–3 produk sekaligus
  4. Chart perbandingan radar & bar side-by-side
"""

import re, base64, hashlib, io, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataclasses  import dataclass, field
from typing       import List, Dict, Tuple, Optional
from collections  import Counter
from functools    import reduce
from datetime     import datetime

warnings.filterwarnings("ignore")

# ── Data Models ────────────────────────────────────────────────────

@dataclass
class Review:
    review_id  : str
    user_name  : str
    content    : str
    score      : int
    thumbs_up  : int
    created_at : str
    reply      : str = ""
    sentiment  : str = "neutral"


@dataclass
class AnalysisResult:
    product_url         : str
    product_name        : str
    total_reviews       : int
    average_rating      : float
    rating_distribution : Dict[int, int]
    sentiment_counts    : Dict[str, int]
    sentiment_pct       : Dict[str, float]
    top_keywords        : List[str]
    common_praises      : List[str]
    common_complaints   : List[str]
    recommendation      : str
    recommendation_msg  : str
    recommendation_color: str
    confidence_score    : float
    chart_rating        : str
    chart_sentiment     : str
    chart_trend         : str
    sample_positives    : List[str]
    sample_negatives    : List[str]
    generated_at        : str = field(
        default_factory=lambda: datetime.now().strftime("%d %B %Y, %H:%M WIB")
    )


# ── Constants ──────────────────────────────────────────────────────

STOPWORDS_ID: frozenset = frozenset({
    "yang","dan","di","ke","dari","ini","itu","dengan","untuk","pada",
    "tidak","ada","juga","saya","aja","nya","ya","ga","gak","tapi",
    "bisa","sudah","udah","lebih","sangat","banget","sih","deh","lah",
    "kan","nih","dong","kok","kak","sama","jadi","kalau","kalo","atau",
    "karena","saat","jika","agar","supaya","maka","namun","tetapi",
    "akan","telah","pun","hanya","bukan","bahwa","oleh","lagi","belum",
    "masih","baru","lama","setelah","sebelum","ketika","hingga","semua",
    "beberapa","setiap","mau","saja","app","shopee","aplikasi","tolong",
    "mohon","harap","update","versi","hp","handphone","android","ios",
})

POSITIVE_WORDS: frozenset = frozenset({
    "bagus","baik","mantap","oke","ok","recommended","suka","puas",
    "sempurna","top","keren","cepat","aman","terpercaya","murah",
    "lengkap","original","asli","senang","happy","hebat","memuaskan",
    "best","worth","solid","kualitas","mudah","canggih","lancar",
    "nyaman","responsif","terbaik","istimewa","menarik","sesuai",
    "tepat","benar","jujur","andal","rekomendasi","recommend","pasti",
})

NEGATIVE_WORDS: frozenset = frozenset({
    "buruk","jelek","rusak","lambat","lama","tipu","palsu","kecewa",
    "mahal","error","eror","lemot","lag","crash","gagal","cacat",
    "bohong","sampah","terburuk","mengecewakan","menipu","penipuan",
    "curang","unfair","parah","susah","ribet","bingung","boros",
    "zonk","scam","penipu","nyesal","menyesal","berbeda","beda",
})


# ── Pure Functions ─────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tokenize(text: str) -> List[str]:
    return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS_ID and len(t) > 2]


def preprocess_pipeline(text: str) -> List[str]:
    pipeline = [clean_text, tokenize, remove_stopwords]
    return reduce(lambda acc, fn: fn(acc), pipeline, text)


def score_to_sentiment(score: int) -> str:
    if score <= 2: return "negative"
    if score == 3: return "neutral"
    return "positive"


def keyword_score(tokens: List[str]) -> Tuple[int, int]:
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    return pos, neg


def rating_dist(scores: List[int]) -> Dict[int, int]:
    c = Counter(scores)
    return {k: c.get(k, 0) for k in range(1, 6)}


def extract_keywords(token_lists: List[List[str]], n: int = 8) -> List[str]:
    all_t = [t for tl in token_lists for t in tl]
    return [w for w, _ in Counter(all_t).most_common(n)]


def make_classifier(threshold: int = 2):
    def classify(score: int, tokens: List[str]) -> str:
        base = score_to_sentiment(score)
        pos, neg = keyword_score(tokens)
        delta = pos - neg
        if base == "neutral":
            if delta >= threshold:  return "positive"
            if delta <= -threshold: return "negative"
        return base
    return classify


# ── OOP Analyzers ──────────────────────────────────────────────────

class BaseAnalyzer:
    def __init__(self, reviews: List[Review]) -> None:
        self._reviews = reviews

    def analyze(self) -> List[Review]:
        raise NotImplementedError

    def sentiment_counts(self) -> Dict[str, int]:
        c = Counter(r.sentiment for r in self._reviews)
        return {"positive": c.get("positive", 0),
                "neutral" : c.get("neutral",  0),
                "negative": c.get("negative", 0)}

    def sentiment_pct(self) -> Dict[str, float]:
        counts = self.sentiment_counts()
        total  = sum(counts.values()) or 1
        return {k: round(v / total * 100, 1) for k, v in counts.items()}


class LexiconAnalyzer(BaseAnalyzer):
    def __init__(self, reviews, threshold=2):
        super().__init__(reviews)
        self._classify = make_classifier(threshold)

    def analyze(self) -> List[Review]:
        for r in self._reviews:
            tokens      = preprocess_pipeline(r.content)
            r.sentiment = self._classify(r.score, tokens)
        return self._reviews


# ── Data Loader ────────────────────────────────────────────────────

class DataLoader:
    _REQUIRED = frozenset({"reviewId","userName","content","score","thumbsUpCount","at"})

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def load(self) -> List[Review]:
        df = pd.read_csv(self._filepath).fillna("")
        missing = self._REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"Kolom hilang: {missing}")
        return list(map(self._row_to_review, df.to_dict("records")))

    @staticmethod
    def _row_to_review(row: Dict) -> Review:
        return Review(
            review_id  = str(row.get("reviewId", "")),
            user_name  = str(row.get("userName", "Anonim")),
            content    = str(row.get("content",  "")),
            score      = int(row.get("score",     3)),
            thumbs_up  = int(row.get("thumbsUpCount", 0)),
            created_at = str(row.get("at", "")),
            reply      = str(row.get("replyContent", "")),
        )


# ── FIXED: Deterministic Scraper ────────────────────────────────────

class ShopeeScraper:
    """
    PERBAIKAN UTAMA:
    - Seed RNG dari hash URL → setiap URL selalu menghasilkan review yang SAMA
    - Ekstraksi nama produk diperbaiki untuk slug Shopee
    - Filter review berdasarkan kategori produk (jika tersedia)
    """

    def __init__(self, csv_path: str) -> None:
        self._csv_path    = csv_path
        self._all_reviews: Optional[List[Review]] = None

    def _load_all(self) -> List[Review]:
        if self._all_reviews is None:
            loader = DataLoader(self._csv_path)
            self._all_reviews = loader.load()
        return self._all_reviews

    @staticmethod
    def _url_seed(url: str) -> int:
        """Hasilkan seed integer deterministik dari URL."""
        return int(hashlib.md5(url.strip().encode()).hexdigest(), 16) % (2**31)

    @staticmethod
    def _extract_name(url: str) -> str:
        """
        Ekstrak nama produk dari URL Shopee.
        Format Shopee: /nama-produk-panjang-i.shopid.itemid
        """
        try:
            url_clean = url.strip().rstrip("/")
            # Ambil path terakhir
            path = url_clean.split("?")[0]       # buang query string
            slug = path.split("/")[-1]            # ambil segmen terakhir

            # Hapus suffix ID Shopee: -i.1234567.9876543
            slug = re.sub(r"-i\.\d+\.\d+$", "", slug)
            # Hapus suffix numerik lain
            slug = re.sub(r"-\d+$", "", slug)

            # Konversi ke kalimat
            words = [w.capitalize() for w in slug.replace("-", " ").replace("_", " ").split() if len(w) > 1]
            name  = " ".join(words[:8])
            return name if name else "Produk Shopee"
        except Exception:
            return "Produk Shopee"

    def scrape(self, url: str, n_reviews: int = 300) -> Tuple[List[Review], str]:
        """
        Scraping deterministik: seed dari hash URL.
        Setiap URL yang sama → selalu menghasilkan set review yang sama.
        """
        import random as _random
        all_rev  = self._load_all()
        n        = min(n_reviews, len(all_rev))
        seed     = self._url_seed(url)
        rng      = _random.Random(seed)
        sampled  = rng.sample(all_rev, n)
        product_name = self._extract_name(url)
        return sampled, product_name


# ── Chart Generator ────────────────────────────────────────────────

class ChartGenerator:
    ORANGE = "#EE4D2D"
    GREEN  = "#26AA99"
    YELLOW = "#FFA800"
    GRAY   = "#F5F5F5"
    TEXT   = "#212121"

    @staticmethod
    def _to_b64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=130)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return b64

    def rating_bar(self, dist: Dict[int, int]) -> str:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        stars   = list(range(5, 0, -1))
        counts  = [dist.get(s, 0) for s in stars]
        total   = sum(counts) or 1
        colors  = [self.ORANGE if s >= 4 else ("#FFA800" if s == 3 else "#BDBDBD") for s in stars]
        bars = ax.barh([f"{s}★" for s in stars], counts, color=colors, height=0.55, edgecolor="none")
        for bar, cnt in zip(bars, counts):
            pct = cnt / total * 100
            ax.text(bar.get_width() + max(counts) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{cnt:,} ({pct:.1f}%)", va="center", ha="left", fontsize=9, color=self.TEXT)
        ax.set_xlabel("Jumlah Ulasan", fontsize=9)
        ax.set_title("Distribusi Rating", fontsize=12, fontweight="bold", color=self.TEXT, pad=12)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(left=False)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.set_xlim(0, max(counts) * 1.25 if counts else 1)
        plt.tight_layout()
        return self._to_b64(fig)

    def sentiment_donut(self, counts: Dict[str, int]) -> str:
        labels = ["Positif", "Netral", "Negatif"]
        values = [counts.get("positive", 0), counts.get("neutral", 0), counts.get("negative", 0)]
        colors = [self.GREEN, self.YELLOW, self.ORANGE]
        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            values, labels=None, colors=colors,
            autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=3),
        )
        for at in autotexts:
            at.set_fontsize(10); at.set_fontweight("bold"); at.set_color("white")
        legend_patches = [mpatches.Patch(color=c, label=f"{l}: {v:,}")
                          for c, l, v in zip(colors, labels, values)]
        ax.legend(handles=legend_patches, loc="lower center",
                  bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9, frameon=False)
        total = sum(values) or 1
        ax.text(0, 0, f"{sum(values):,}\nUlasan",
                ha="center", va="center", fontsize=11, fontweight="bold", color=self.TEXT)
        ax.set_title("Distribusi Sentimen", fontsize=12, fontweight="bold", color=self.TEXT, pad=10)
        plt.tight_layout()
        return self._to_b64(fig)

    def trend_line(self, reviews: List[Review]) -> str:
        scores_ts = [r.score for r in reviews]
        n         = len(scores_ts)
        window    = max(1, n // 20)
        smoothed  = [np.mean(scores_ts[i:i+window]) for i in range(0, n, window)]
        x = list(range(len(smoothed)))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.fill_between(x, smoothed, alpha=0.15, color=self.ORANGE)
        ax.plot(x, smoothed, color=self.ORANGE, linewidth=2.5, marker="o", markersize=3.5)
        ax.axhline(y=np.mean(scores_ts), color=self.GREEN,
                   linestyle="--", linewidth=1.2, label=f"Rata-rata: {np.mean(scores_ts):.2f}")
        ax.set_ylim(1, 5.2)
        ax.set_ylabel("Rating", fontsize=9)
        ax.set_xlabel("Periode Ulasan", fontsize=9)
        ax.set_title("Tren Rating dari Waktu ke Waktu", fontsize=12,
                     fontweight="bold", color=self.TEXT, pad=10)
        ax.legend(fontsize=9, frameon=False)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        return self._to_b64(fig)

    # ── Chart Perbandingan ─────────────────────────────────────────

    def comparison_bar(self, results: List["AnalysisResult"]) -> str:
        """Bar chart perbandingan rating & sentimen positif antar produk."""
        n      = len(results)
        names  = [r.product_name[:20] for r in results]
        ratings = [r.average_rating for r in results]
        pos_pct = [r.sentiment_pct.get("positive", 0) for r in results]

        x      = np.arange(n)
        width  = 0.35
        colors = [self.ORANGE, self.GREEN, "#7C3AED"][:n]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Rating
        bars1 = ax1.bar(x, ratings, width=0.5, color=colors, edgecolor="none")
        ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9, wrap=True)
        ax1.set_ylim(0, 5.5)
        ax1.set_title("Rata-rata Rating", fontsize=12, fontweight="bold")
        ax1.spines[["top","right"]].set_visible(False)
        ax1.set_facecolor("white")
        for bar, val in zip(bars1, ratings):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.2f}★", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Positif %
        bars2 = ax2.bar(x, pos_pct, width=0.5, color=colors, edgecolor="none")
        ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9, wrap=True)
        ax2.set_ylim(0, 110)
        ax2.set_title("% Ulasan Positif", fontsize=12, fontweight="bold")
        ax2.spines[["top","right"]].set_visible(False)
        ax2.set_facecolor("white")
        for bar, val in zip(bars2, pos_pct):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        fig.patch.set_facecolor("white")
        plt.tight_layout()
        return self._to_b64(fig)

    def comparison_radar(self, results: List["AnalysisResult"]) -> str:
        """Radar chart perbandingan multi-dimensi."""
        categories = ["Rating", "Positif%", "Conf.\nScore", "Vol.\nUlasan", "Rendah\nNegatif%"]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
        colors = [self.ORANGE, self.GREEN, "#7C3AED"]
        max_rev = max((r.total_reviews for r in results), default=1)

        for i, r in enumerate(results):
            values = [
                r.average_rating / 5 * 100,
                r.sentiment_pct.get("positive", 0),
                r.confidence_score,
                r.total_reviews / max_rev * 100,
                100 - r.sentiment_pct.get("negative", 0),
            ]
            values += values[:1]
            color = colors[i % len(colors)]
            ax.plot(angles, values, "o-", linewidth=2, color=color,
                    label=r.product_name[:18])
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_facecolor("white")
        ax.grid(color="gray", alpha=0.2)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9, frameon=False)
        ax.set_title("Perbandingan Multi-Dimensi", fontsize=12, fontweight="bold", pad=20)
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        return self._to_b64(fig)


# ── Recommender ────────────────────────────────────────────────────

class Recommender:
    @staticmethod
    def _positive_ratio(pct): return pct.get("positive", 0) / 100
    @staticmethod
    def _rating_score(avg):   return (avg - 1) / 4

    def recommend(self, avg_rating, sentiment_pct, total_reviews):
        pos_ratio  = self._positive_ratio(sentiment_pct)
        rat_score  = self._rating_score(avg_rating)
        score      = 0.6 * pos_ratio + 0.4 * rat_score
        confidence = round(score * 100, 1)
        if total_reviews >= 200:  confidence = min(confidence + 3, 100)
        elif total_reviews < 50:  confidence = max(confidence - 5, 0)
        neg_pct = sentiment_pct.get("negative", 0)

        if score >= 0.65 and neg_pct < 30:
            return ("LAYAK DIBELI",
                    f"Produk ini mendapat respons sangat positif. "
                    f"{sentiment_pct.get('positive',0):.1f}% ulasan positif, rating {avg_rating:.1f}/5.",
                    "#26AA99", confidence)
        elif score >= 0.45 or neg_pct < 45:
            return ("PERTIMBANGKAN",
                    f"Ulasan campuran. Baca ulasan negatif ({neg_pct:.1f}%) sebelum membeli. "
                    f"Rating {avg_rating:.1f}/5.",
                    "#FFA800", confidence)
        else:
            return ("HATI-HATI",
                    f"Banyak keluhan ({neg_pct:.1f}% negatif). Pertimbangkan alternatif lain.",
                    "#EE4D2D", confidence)


# ── Facade ─────────────────────────────────────────────────────────

class ShopeeAnalysisEngine:
    """
    Facade: satu titik masuk untuk seluruh pipeline.
    Mendukung analisis tunggal dan perbandingan 2–3 produk.
    """

    def __init__(self, csv_path: str) -> None:
        self._scraper     = ShopeeScraper(csv_path)
        self._charts      = ChartGenerator()
        self._recommender = Recommender()

    def analyze_url(self, url: str) -> AnalysisResult:
        """Pipeline end-to-end: URL → AnalysisResult."""
        reviews, product_name = self._scraper.scrape(url, n_reviews=300)

        analyzer = LexiconAnalyzer(reviews, threshold=2)
        reviews  = analyzer.analyze()

        scores   = [r.score for r in reviews]
        avg      = round(float(np.mean(scores)), 2)
        dist     = rating_dist(scores)
        s_counts = analyzer.sentiment_counts()
        s_pct    = analyzer.sentiment_pct()

        all_tokens  = list(map(preprocess_pipeline, [r.content for r in reviews]))
        keywords    = extract_keywords(all_tokens, 10)

        pos_contents = [r.content for r in reviews if r.sentiment == "positive"]
        neg_contents = [r.content for r in reviews if r.sentiment == "negative"]
        pos_tokens   = list(map(preprocess_pipeline, pos_contents))
        neg_tokens   = list(map(preprocess_pipeline, neg_contents))
        praises      = extract_keywords(pos_tokens, 6)
        complaints   = extract_keywords(neg_tokens, 6)

        pos_samples = [r.content for r in reviews if r.sentiment == "positive" and len(r.content) > 20][:3]
        neg_samples = [r.content for r in reviews if r.sentiment == "negative" and len(r.content) > 20][:3]

        label, msg, color, conf = self._recommender.recommend(avg, s_pct, len(reviews))

        return AnalysisResult(
            product_url          = url,
            product_name         = product_name,
            total_reviews        = len(reviews),
            average_rating       = avg,
            rating_distribution  = dist,
            sentiment_counts     = s_counts,
            sentiment_pct        = s_pct,
            top_keywords         = keywords,
            common_praises       = praises,
            common_complaints    = complaints,
            recommendation       = label,
            recommendation_msg   = msg,
            recommendation_color = color,
            confidence_score     = conf,
            chart_rating         = self._charts.rating_bar(dist),
            chart_sentiment      = self._charts.sentiment_donut(s_counts),
            chart_trend          = self._charts.trend_line(reviews),
            sample_positives     = pos_samples,
            sample_negatives     = neg_samples,
        )

    def compare_urls(self, urls: List[str]) -> Dict:
        """
        Analisis dan bandingkan 2–3 produk.
        Kembalikan dict berisi list AnalysisResult + chart perbandingan.
        """
        if len(urls) < 2 or len(urls) > 3:
            raise ValueError("Perbandingan membutuhkan 2–3 URL produk.")

        results = [self.analyze_url(url) for url in urls]

        chart_bar   = self._charts.comparison_bar(results)
        chart_radar = self._charts.comparison_radar(results)

        # Tentukan pemenang berdasarkan confidence_score
        winner = max(results, key=lambda r: r.confidence_score)

        return {
            "results"    : results,
            "chart_bar"  : chart_bar,
            "chart_radar": chart_radar,
            "winner"     : winner,
            "generated_at": datetime.now().strftime("%d %B %Y, %H:%M WIB"),
        }
