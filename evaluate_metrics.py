"""
evaluate_metrics.py — Skrip Evaluasi Metrik Akurasi Akurat untuk Paper Ilmiah
Mengukur performa pencarian informasi menggunakan P@K, R@K, dan MRR.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from cv_engine import CVAnalyzer

# 1. Definisi Dataset Uji (Mock CVs + Ground Truth Roles)
# Representasi CV yang realistis beserta label kebenaran (ground truth)
test_cases = [
    {
        "cv_text": "Saya adalah Data Scientist berpengalaman dalam mengolah data besar menggunakan Python, SQL, Pandas, Scikit-learn, dan melatih model Machine Learning.",
        "ground_truth": ["Data Scientist", "Data Analyst", "Machine Learning Engineer"]
    },
    {
        "cv_text": "Seorang Frontend Developer ahli membangun antarmuka web responsif dengan React.js, Vue.js, Tailwind CSS, HTML, CSS, dan JavaScript modern.",
        "ground_truth": ["Frontend Developer", "Fullstack Developer", "Product Designer"]
    },
    {
        "cv_text": "Software Engineer fokus pada backend development dengan Node.js, Go, PostgreSQL, MongoDB, pembuatan RESTful API, dan arsitektur microservices.",
        "ground_truth": ["Backend Developer", "Fullstack Developer", "Database Administrator"]
    },
    {
        "cv_text": "UI/UX Designer terbiasa merancang wireframe, prototype interaktif menggunakan Figma, Adobe XD, dan melakukan Usability Testing kepada pengguna.",
        "ground_truth": ["UI/UX Designer", "Product Designer", "UX Researcher"]
    },
    {
        "cv_text": "DevOps Engineer berpengalaman mengelola CI/CD pipeline dengan Jenkins, Terraform, containerisasi Docker, Kubernetes, dan cloud computing AWS/GCP.",
        "ground_truth": ["DevOps Engineer", "Cloud Architect", "Site Reliability Engineer (SRE)"]
    },
    {
        "cv_text": "Cyber Security Analyst fokus pada keamanan jaringan, SIEM, penetration testing menggunakan Kali Linux, Metasploit, dan audit ISO 27001.",
        "ground_truth": ["Cyber Security Analyst", "Penetration Tester (Ethical Hacker)", "Security Engineer"]
    },
    {
        "cv_text": "Technical Recruiter berpengalaman mencari talenta IT, melakukan wawancara, manajemen HR, sourcing LinkedIn Recruiter, dan Employee Relations.",
        "ground_truth": ["Technical Recruiter", "HR Generalist", "HR Manager"]
    },
    {
        "cv_text": "Digital Marketing Specialist dengan keahlian riset kata kunci SEO, Google Analytics, social media management, copywriting iklan, dan Google Ads.",
        "ground_truth": ["Digital Marketing Specialist", "SEO Specialist", "Social Media Manager"]
    }
]

def calculate_metrics(recommendations, ground_truth, k):
    """Menghitung Precision@K dan Recall@K untuk satu kueri."""
    top_k_recs = recommendations[:k]
    matched = [rec for rec in top_k_recs if rec in ground_truth]
    
    precision = len(matched) / k
    recall = len(matched) / len(ground_truth) if len(ground_truth) > 0 else 0.0
    return precision, recall

def calculate_reciprocal_rank(recommendations, ground_truth):
    """Menghitung Reciprocal Rank untuk satu kueri."""
    for rank, rec in enumerate(recommendations, start=1):
        if rec in ground_truth:
            return 1.0 / rank
    return 0.0

def main():
    print("=" * 70)
    print("MEMULAI EVALUASI METRIK AKURASI MODEL RESUMATE AI")
    print("=" * 70)
    
    # Inisialisasi Analyzer
    csv_path = "job_roles_dataset.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} tidak ditemukan! Jalankan generate_jobs.py terlebih dahulu.")
        return
        
    analyzer = CVAnalyzer(csv_path)
    
    # Validasi apakah SBERT aktif
    sbert_ready = analyzer.job_sbert is not None
    if sbert_ready:
        print("[STATUS] Sentence-BERT (SBERT) API Aktif. Mengevaluasi 3 Model.")
    else:
        print("[WARNING] SBERT API tidak aktif (fallback). Pengujian SBERT akan dilewati.")
    
    # Data storage untuk hasil
    methods = ["TF-IDF (Leksikal)", "LSA (Semantik Klasik)", "Hybrid (TF-IDF + SBERT)"]
    if not sbert_ready:
        methods = ["TF-IDF (Leksikal)", "LSA (Semantik Klasik)"]
        
    results = {method: {"P@1": [], "P@3": [], "P@5": [], "R@1": [], "R@3": [], "R@5": [], "MRR": []} for method in methods}
    
    for case in test_cases:
        cv_text = case["cv_text"]
        gt = case["ground_truth"]
        
        # Ekstraksi vektor
        cv_tfidf = analyzer.vectorizer.transform([cv_text])
        cv_semantic = analyzer.lsa_model.transform(cv_tfidf)
        
        # 1. Evaluasi TF-IDF
        tfidf_sim = cosine_similarity(cv_tfidf, analyzer.job_tfidf)[0]
        tfidf_recs = [analyzer.jobs_df.iloc[idx]["role"] for idx in tfidf_sim.argsort()[::-1]]
        
        # 2. Evaluasi LSA
        lsa_sim = cosine_similarity(cv_semantic, analyzer.job_semantic)[0]
        lsa_recs = [analyzer.jobs_df.iloc[idx]["role"] for idx in lsa_sim.argsort()[::-1]]
        
        # 3. Evaluasi Hybrid
        hybrid_recs = None
        if sbert_ready:
            cv_sbert = analyzer._get_sbert_embeddings([cv_text])
            if cv_sbert is not None:
                sbert_sim = cosine_similarity(cv_sbert, analyzer.job_sbert)[0]
                hybrid_sim = 0.4 * tfidf_sim + 0.6 * sbert_sim
                hybrid_recs = [analyzer.jobs_df.iloc[idx]["role"] for idx in hybrid_sim.argsort()[::-1]]
        
        # Hitung metrik untuk masing-masing metode
        recs_map = {
            "TF-IDF (Leksikal)": tfidf_recs,
            "LSA (Semantik Klasik)": lsa_recs
        }
        if hybrid_recs:
            recs_map["Hybrid (TF-IDF + SBERT)"] = hybrid_recs
            
        for method, recs in recs_map.items():
            for k in [1, 3, 5]:
                p, r = calculate_metrics(recs, gt, k)
                results[method][f"P@{k}"].append(p)
                results[method][f"R@{k}"].append(r)
            results[method]["MRR"].append(calculate_reciprocal_rank(recs, gt))
            
    # Rekapitulasi Rata-rata Metrik
    report_data = []
    for method in methods:
        row = {"Model/Metode": method}
        for k in [1, 3, 5]:
            row[f"P@{k}"] = f"{np.mean(results[method][f'P@{k}']) * 100:.2f}%"
            row[f"R@{k}"] = f"{np.mean(results[method][f'R@{k}']) * 100:.2f}%"
        row["MRR"] = f"{np.mean(results[method]['MRR']):.4f}"
        report_data.append(row)
        
    headers = ["Model/Metode", "P@1", "P@3", "P@5", "R@1", "R@3", "R@5", "MRR"]
    header_str = " | ".join(headers)
    separator = " | ".join(["---"] * len(headers))
    print(header_str)
    print(separator)
    for row in report_data:
        row_str = " | ".join([str(row[h]) for h in headers])
        print(row_str)
    print("=" * 80)
    print("\nTips Penulisan Paper:")
    print("1. Nilai Precision@1 (P@1) yang tinggi menunjukkan rekomendasi peringkat ke-1 sangat akurat.")
    print("2. Nilai MRR mendekati 1.000 menunjukkan kecocokan utama selalu berada di peringkat teratas.")
    print("3. Gunakan tabel di atas di Bab IV (Hasil & Pembahasan) untuk membuktikan keunggulan model Hybrid Anda.")

if __name__ == "__main__":
    main()
