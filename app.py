"""
app.py — CV-Match AI (Sains Data Practicum)
"""

import os, uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from cv_engine import CVAnalyzer

app = Flask(__name__)
app.secret_key = "cv-match-ai-2026-super-secret"

# Setup Upload Folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 # Max 5MB PDF

# Init Engine
CSV_PATH = os.path.join(os.path.dirname(__file__), "job_roles_dataset.csv")
analyzer = CVAnalyzer(CSV_PATH)

# In-memory cache for results
_cache = {}

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'cv_file' not in request.files:
            flash('Tidak ada file yang diunggah', 'error')
            return redirect(request.url)
            
        file = request.files['cv_file']
        if file.filename == '':
            flash('File tidak dipilih', 'error')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())
            safe_filename = f"{unique_id}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            file.save(filepath)
            
            # Proses CV
            try:
                result = analyzer.analyze_cv(filepath, top_n=5)
                
                # Cek jika ada error dari engine (misal teks gagal diekstrak)
                if "error" in result:
                    flash(result["error"], "error")
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return redirect(request.url)
                
                # Simpan ke cache
                _cache[unique_id] = {
                    "filename": filename,
                    "matches": result.get("matches", []),
                    "quality": result.get("quality", {}),
                    "soft_skills": result.get("soft_skills", []),
                    "keywords": result.get("keywords", [])
                }
                
                # Hapus file PDF setelah diproses
                os.remove(filepath)
                
                return redirect(url_for('result', session_id=unique_id))
            except Exception as e:
                flash(f'Terjadi kesalahan saat memproses: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Hanya format PDF yang diperbolehkan', 'error')
            return redirect(request.url)
            
    return render_template("home.html")

@app.route("/hasil/<session_id>")
def result(session_id):
    if session_id not in _cache:
        flash("Sesi telah berakhir atau tidak valid. Silakan unggah ulang.", "error")
        return redirect(url_for("index"))
        
    data = _cache[session_id]
    return render_template("hasil.html", 
                           filename=data['filename'],
                           matches=data['matches'],
                           quality=data['quality'],
                           soft_skills=data['soft_skills'],
                           keywords=data['keywords'])

if __name__ == "__main__":
    app.run(debug=True, port=5000)
