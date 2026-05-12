"""
app.py — Flask Web Application v2.0
Shopee Customer Sentiment Analysis + Perbandingan Produk
"""

import os, uuid, re
from flask import Flask, render_template, request, redirect, url_for, jsonify
from sentiment_engine import ShopeeAnalysisEngine

app        = Flask(__name__)
app.secret_key = "shopee-sentiment-2025-telkom-v2"

CSV_PATH = os.path.join(os.path.dirname(__file__), "Shopee_Sampled_Reviews.csv")
engine   = ShopeeAnalysisEngine(CSV_PATH)

# In-memory cache (production: Redis)
_cache: dict = {}


def is_valid_url(url: str) -> bool:
    """Validasi: harus ada karakter URL minimal."""
    return bool(url and len(url.strip()) >= 5)


# ── Routes ─────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/analisis", methods=["GET", "POST"])
def analisis():
    error = None
    if request.method == "POST":
        url = request.form.get("product_url", "").strip()
        if not url:
            error = "URL produk tidak boleh kosong."
        elif not is_valid_url(url):
            error = "Masukkan URL produk yang valid."
        else:
            try:
                result     = engine.analyze_url(url)
                sid        = str(uuid.uuid4())
                _cache[sid] = result
                return redirect(url_for("hasil", sid=sid))
            except Exception as e:
                error = f"Terjadi kesalahan: {str(e)}"
    return render_template("analisis.html", error=error)


@app.route("/hasil/<sid>")
def hasil(sid):
    result = _cache.get(sid)
    if not result:
        return redirect(url_for("analisis"))
    return render_template("hasil.html", r=result)


@app.route("/bandingkan", methods=["GET", "POST"])
def bandingkan():
    error = None
    if request.method == "POST":
        urls = [
            request.form.get("url1", "").strip(),
            request.form.get("url2", "").strip(),
            request.form.get("url3", "").strip(),
        ]
        # Filter URL kosong
        urls = [u for u in urls if u]

        if len(urls) < 2:
            error = "Masukkan minimal 2 URL produk untuk perbandingan."
        else:
            try:
                comparison = engine.compare_urls(urls)
                sid        = str(uuid.uuid4())
                _cache[sid] = comparison
                return redirect(url_for("hasil_bandingkan", sid=sid))
            except Exception as e:
                error = f"Terjadi kesalahan: {str(e)}"
    return render_template("bandingkan.html", error=error)


@app.route("/hasil-bandingkan/<sid>")
def hasil_bandingkan(sid):
    data = _cache.get(sid)
    if not data:
        return redirect(url_for("bandingkan"))
    return render_template("hasil_bandingkan.html", data=data)


@app.route("/tentang")
def tentang():
    return render_template("tentang.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
