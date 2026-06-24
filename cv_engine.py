"""
cv_engine.py — Core Engine for CV-Match AI (ADVANCED VERSION)

ALUR PEMROSESAN DATA (DATA PIPELINE):
[TAHAP 1] Data Gathering     : Membaca Dataset Lowongan (CSV) dan CV pengguna (PDF).
[TAHAP 2] Data Preprocessing : Membersihkan teks dari karakter aneh, angka, dan mengubah ke huruf kecil.
[TAHAP 3] Feature Extraction : Mengubah teks menjadi representasi angka menggunakan TF-IDF.
[TAHAP 4] Semantic Modeling  : Menerapkan LSA (SVD) untuk memahami makna konteks kata secara matematis.
[TAHAP 5] Expert System      : Menerapkan Rule-Based NLP untuk mendeteksi Soft Skills & Kualitas CV.
"""

import os, re
import PyPDF2
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class JobMatch:
    job_title: str
    category: str
    similarity_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    description: str
    required_skills_raw: str
    improvement_advice: str
    radar_labels: List[str] 
    radar_data: List[int]   


class CVAnalyzer:
    def __init__(self, dataset_path: str):
        # ==========================================
        # [TAHAP 1: DATA GATHERING]
        # ==========================================
        self.dataset_path = dataset_path
        self.jobs_df = pd.read_csv(dataset_path)
        
        # Inisialisasi Vectorizer dengan Stopwords
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2))
        self.lsa_model = TruncatedSVD(n_components=20, random_state=42)
        
        self._prepare_dataset()

    def _prepare_dataset(self):
        """Menyiapkan model dari dataset lowongan."""
        self.jobs_df['corpus'] = self.jobs_df['role'] + " " + \
                                 self.jobs_df['category'] + " " + \
                                 self.jobs_df['desc'] + " " + \
                                 self.jobs_df['skills']
        
        # ==========================================
        # [TAHAP 3: FEATURE EXTRACTION]
        # Mengubah Teks menjadi Matriks Angka (TF-IDF)
        # ==========================================
        self.job_tfidf = self.vectorizer.fit_transform(self.jobs_df['corpus'].fillna(""))
        
        # ==========================================
        # [TAHAP 4: SEMANTIC MODELING]
        # Mengompresi dimensi matriks agar AI paham konteks/makna (LSA)
        # ==========================================
        n_comp = min(20, self.job_tfidf.shape[1] - 1)
        self.lsa_model = TruncatedSVD(n_components=n_comp, random_state=42)
        self.job_semantic = self.lsa_model.fit_transform(self.job_tfidf)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        ==========================================
        [TAHAP 2: DATA PREPROCESSING]
        Ekstraksi dan Pembersihan Teks
        ==========================================
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + " "
        except Exception as e:
            return ""
        
        # Membersihkan simbol khusus dengan Regex
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\+\#\-]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def extract_soft_skills(self, cv_text: str) -> List[Dict]:
        """
        ==========================================
        [TAHAP 5A: EXPERT SYSTEM - SOFT SKILL PROFILING]
        Menganalisis kepribadian dari pilihan kata (Lexicon-based)
        ==========================================
        """
        text = cv_text.lower()
        
        # Kamus (Lexicon) Kata Kunci untuk setiap Soft Skill
        soft_skills_lexicon = {
            "Leadership (Kepemimpinan)": ["lead", "manage", "mentor", "memimpin", "mengelola", "head", "director", "manager", "koordinator"],
            "Teamwork (Kerjasama)": ["collaborate", "team", "support", "kolaborasi", "tim", "bersama", "assist", "membantu", "cooperate"],
            "Communication (Komunikasi)": ["present", "communicate", "negotiate", "komunikasi", "presentasi", "negosiasi", "speaker", "menulis"],
            "Problem Solving (Penyelesaian Masalah)": ["solve", "analyze", "resolve", "troubleshoot", "analisis", "menyelesaikan", "solusi", "mengatasi"],
            "Adaptability (Adaptasi)": ["adapt", "flexible", "agile", "adaptasi", "fleksibel", "dinamis", "cepat belajar", "fast learner"]
        }
        
        detected_soft_skills = []
        for skill_name, keywords in soft_skills_lexicon.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > 0:
                # Menghitung level berdasarkan frekuensi kata
                level = "Tinggi" if count >= 3 else "Menengah" if count == 2 else "Dasar"
                detected_soft_skills.append({
                    "name": skill_name,
                    "level": level,
                    "count": count
                })
                
        # Urutkan berdasarkan yang paling kuat
        return sorted(detected_soft_skills, key=lambda x: x['count'], reverse=True)

    def evaluate_cv_quality(self, cv_text: str) -> Dict:
        """
        ==========================================
        [TAHAP 5B: EXPERT SYSTEM - QUALITY SCORER]
        ==========================================
        """
        text = cv_text.lower()
        
        metrics = re.findall(r'(\d+%)|(rp\s*\d+)|(\$\d+)|(\d+\+)|(\d+ (juta|ribu|million|k))', text)
        metric_score = min(len(metrics) * 10, 40) 
        
        action_verbs = ['led', 'memimpin', 'developed', 'mengembangkan', 'created', 'membuat', 
                        'managed', 'mengelola', 'increased', 'meningkatkan', 'achieved', 'mencapai',
                        'designed', 'merancang', 'optimized', 'mengoptimalkan']
        
        verb_count = sum(1 for verb in action_verbs if verb in text)
        verb_score = min(verb_count * 5, 40) 
        
        structure_score = 0
        if 'education' in text or 'pendidikan' in text or 'university' in text or 'universitas' in text:
            structure_score += 10
        if 'experience' in text or 'pengalaman' in text or 'work' in text:
            structure_score += 10
            
        total_score = metric_score + verb_score + structure_score
        
        advice = []
        if metric_score < 20:
            advice.append("CV kurang memuat angka pencapaian (metrik). Contoh: Ubah 'Membuat program' menjadi 'Membuat program yang meningkatkan efisiensi 30%'.")
        if verb_score < 15:
            advice.append("Gunakan lebih banyak kata kerja aktif (Action Verbs) seperti 'Menginisiasi', 'Memimpin', untuk menunjukkan inisiatif.")
        if structure_score < 20:
            advice.append("Pastikan memiliki bagian 'Pendidikan/Education' dan 'Pengalaman/Experience' yang jelas.")
            
        if not advice:
            advice.append("Luar Biasa! CV Anda sudah berorientasi pada pencapaian dengan metrik yang jelas.")

        return {
            "score": total_score,
            "metric_count": len(metrics),
            "verb_count": verb_count,
            "advice": advice
        }

    def generate_advice(self, role: str, missing: List[str], match_score: float) -> str:
        if match_score >= 80 and len(missing) <= 1:
            return "CV Anda secara semantik sudah sangat kuat untuk posisi ini! Bersiaplah untuk wawancara teknis."
        if len(missing) == 0:
            return "Kata kunci Anda sempurna! Algoritma mendeteksi kecocokan penuh dengan kebutuhan industri."
            
        advice = f"Untuk peran {role}, Anda perlu mempelajari keahlian ini: {', '.join(missing[:3])}. "
        if match_score < 40:
            advice += "Sistem LSA (Semantic AI) mendeteksi gap yang besar. Bangun proyek portofolio baru!"
        else:
            advice += "Semantic engine melihat Anda punya dasar yang baik, tapi istilah teknis harus lebih eksplisit di CV."
        return advice

    def analyze_cv(self, pdf_path: str, top_n: int = 5) -> Dict:
        """Fungsi Utama yang Menyatukan Seluruh Pipeline"""
        
        # [TAHAP 2] Preprocessing
        cv_text = self.extract_text_from_pdf(pdf_path)
        if not cv_text:
            return {"error": "Gagal membaca teks dari PDF"}

        # [TAHAP 5A & 5B] Soft Skills & CV Quality
        quality_report = self.evaluate_cv_quality(cv_text)
        soft_skills_report = self.extract_soft_skills(cv_text)

        # [TAHAP 3 & 4] Feature Extraction & Semantic Matching
        cv_tfidf = self.vectorizer.transform([cv_text])
        cv_semantic = self.lsa_model.transform(cv_tfidf)
        
        # Hitung Similarity menggunakan hasil Semantik (LSA)
        similarities = cosine_similarity(cv_semantic, self.job_semantic)[0]
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        matches = []
        for idx in top_indices:
            raw_score = float(similarities[idx])
            score = max(0, min(100, (raw_score * 100) + 20)) # Normalisasi score
                
            row = self.jobs_df.iloc[idx]
            req_skills = [s.strip() for s in str(row['skills']).split(',')]
            
            matched = []
            missing = []
            cv_lower = cv_text.lower()
            
            radar_labels = []
            radar_data = []
            
            for skill in req_skills:
                if len(radar_labels) < 6:
                    radar_labels.append(skill[:15]) 
                
                is_match = bool(re.search(r'\b' + re.escape(skill.lower()) + r'\b', cv_lower) or skill.lower() in cv_lower)
                
                if is_match:
                    matched.append(skill)
                    if len(radar_labels) <= 6: radar_data.append(10)
                else:
                    missing.append(skill)
                    if len(radar_labels) <= 6: radar_data.append(2)
            
            while len(radar_labels) < 3:
                radar_labels.append("General Skill")
                radar_data.append(5)
            
            advice = self.generate_advice(str(row['role']), missing, score)
            
            matches.append(JobMatch(
                job_title=str(row['role']),
                category=str(row['category']),
                similarity_score=round(score, 1),
                matched_skills=matched,
                missing_skills=missing,
                description=str(row['desc']),
                required_skills_raw=str(row['skills']),
                improvement_advice=advice,
                radar_labels=radar_labels,
                radar_data=radar_data
            ))
            
        return {
            "quality": quality_report,
            "soft_skills": soft_skills_report,
            "matches": matches,
            "keywords": self.extract_keywords_from_cv(cv_text)
        }

    def extract_keywords_from_cv(self, text: str) -> List[str]:
        vec = TfidfVectorizer(stop_words='english', max_features=10)
        try:
            vec.fit([text])
            return list(vec.get_feature_names_out())
        except:
            return []
