"""
cv_engine.py — Core Engine for ResuMate AI (ADVANCED VERSION)

ALUR PEMROSESAN DATA (DATA PIPELINE):
[TAHAP 1] Data Gathering     : Membaca Dataset Lowongan (CSV) dan CV pengguna (PDF).
[TAHAP 2] Data Preprocessing : Membersihkan teks dari karakter aneh, angka, dan mengubah ke huruf kecil.
[TAHAP 3] Feature Extraction : Mengubah teks menjadi representasi angka menggunakan TF-IDF.
[TAHAP 4] Semantic Modeling  : Menerapkan LSA (SVD) untuk memahami makna konteks kata secara matematis.
[TAHAP 5] Expert System      : Menerapkan Rule-Based NLP untuk mendeteksi Soft Skills & Kualitas CV.
"""

# cv_engine.py — Core Engine for ResuMate AI (ADVANCED VERSION WITH EXTENDED FEATURES)

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
    def __init__(self, dataset_path: str, hf_token: str = None):
        self.dataset_path = dataset_path
        self.jobs_df = pd.read_csv(dataset_path)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        
        # Gabungan Stopwords Bahasa Inggris dan Bahasa Indonesia
        self.id_stopwords = ['dan', 'di', 'ke', 'dari', 'yang', 'untuk', 'pada', 'dengan', 
                             'adalah', 'sebagai', 'ini', 'itu', 'saya', 'kami', 'telah', 'dalam']
        custom_stopwords = list(TfidfVectorizer(stop_words='english').get_stop_words()) + self.id_stopwords
        
        self.vectorizer = TfidfVectorizer(stop_words=custom_stopwords, lowercase=True, ngram_range=(1, 2))
        self._prepare_dataset()

    def _get_sbert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Mendapatkan embedding SBERT dari Hugging Face Inference API."""
        if not self.hf_token:
            return None
        
        api_url = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        import requests
        import time
        try:
            response = requests.post(api_url, headers=headers, json={"inputs": texts}, timeout=25)
            if response.status_code == 200:
                embeddings = response.json()
                if isinstance(embeddings, list) and len(embeddings) > 0:
                    return np.array(embeddings)
            elif response.status_code == 503:
                # Model loading, tunggu 5 detik dan coba lagi
                time.sleep(5)
                response = requests.post(api_url, headers=headers, json={"inputs": texts}, timeout=25)
                if response.status_code == 200:
                    embeddings = response.json()
                    if isinstance(embeddings, list) and len(embeddings) > 0:
                        return np.array(embeddings)
        except Exception as e:
            pass
        return None

    def _prepare_dataset(self):
        self.jobs_df['corpus'] = self.jobs_df['role'] + " " + \
                                 self.jobs_df['category'] + " " + \
                                 self.jobs_df['desc'] + " " + \
                                 self.jobs_df['skills']
        
        self.job_tfidf = self.vectorizer.fit_transform(self.jobs_df['corpus'].fillna(""))
        
        n_samples, n_features = self.job_tfidf.shape
        n_comp = min(30, n_features - 1, n_samples - 1)
        
        self.lsa_model = TruncatedSVD(n_components=n_comp, random_state=42)
        self.job_semantic = self.lsa_model.fit_transform(self.job_tfidf)
        
        # Pre-calculate SBERT embeddings untuk seluruh lowongan kerja
        self.job_sbert = None
        if self.hf_token:
            corpora = self.jobs_df['corpus'].fillna("").tolist()
            embeddings = self._get_sbert_embeddings(corpora)
            if embeddings is not None:
                self.job_sbert = embeddings

    def extract_text_from_pdf(self, pdf_path: str) -> str:
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
        
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\+\#\-]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    # =========================================================================
    # [FITUR 1] MARKET INSIGHTS DASHBOARD
    # =========================================================================
    def get_market_insights(self, category: str = None) -> Dict:
        """Mengekstrak tren keahlian populer dari dataset untuk kebutuhan statistik pasar kerja."""
        df_filtered = self.jobs_df if not category else self.jobs_df[self.jobs_df['category'].str.lower() == category.lower()]
        
        all_skills = []
        for skills_raw in df_filtered['skills'].dropna():
            skills_list = [s.strip() for s in skills_raw.split(',')]
            all_skills.extend(skills_list)
            
        if not all_skills:
            return {"category": category or "Semua", "top_skills": {}}
            
        # Mengambil 8 keahlian tertinggi yang paling banyak dicari industri
        skill_counts = pd.Series(all_skills).value_counts().head(8).to_dict()
        return {
            "category": category if category else "Semua Kategori",
            "top_skills": skill_counts
        }

    # =========================================================================
    # [FITUR 2] AI RESUME TAILOR (SARAN TRANSFORMASI KALIMAT)
    # =========================================================================
    def generate_resume_tailor_suggestions(self, cv_text: str) -> List[Dict]:
        """Mengidentifikasi frasa/kalimat lemah di CV dan memberikan saran alternatif berbasis hasil."""
        text_lower = cv_text.lower()
        suggestions = []
        
        # Pemetaan pola kalimat pasif/biasa ke rekomendasi action-verbs kuantitatif
        weak_patterns = {
            "membuat program": "Mengembangkan sistem otomasi berbasis [Teknologi] yang berhasil meningkatkan efisiensi pemrosesan data sebesar 25%.",
            "bertanggung jawab atas": "Memimpin koordinasi tim proyek dalam mengelola deployment aplikasi dengan tingkat keberhasilan penyelesaian 100%.",
            "mempelajari data": "Menganalisis data berskala besar untuk mengekstrak insight bisnis terstruktur yang mendukung pengambilan keputusan strategis.",
            "membantu tugas": "Berkolaborasi aktif lintas fungsi untuk menyelenggarakan program edukasi media interaktif yang menjangkau target audiens.",
            "bisa pemrograman": "Mengimplementasikan penulisan kode yang bersih (clean code) dan arsitektur terstruktur untuk mengoptimalkan kinerja sistem."
        }
        
        for weak_phrase, recommendation in weak_patterns.items():
            if weak_phrase in text_lower:
                suggestions.append({
                    "weak_phrase": weak_phrase,
                    "recommended_transformation": recommendation
                })
                
        if not suggestions:
            suggestions.append({
                "weak_phrase": "Format Kualitatif Umum",
                "recommended_transformation": "Gunakan formula 'X-Y-Z' (Mencapai target [X], diukur dengan metrik [Y], melalui tindakan konstruktif [Z]) untuk penulisan poin pengalaman."
            })
            
        return suggestions

    # =========================================================================
    # [FITUR 3] AUTOMATIC RESUME PARSER (AUTO-FILL FORM DATA)
    # =========================================================================
    def parse_basic_info(self, cv_text: str) -> Dict:
        """Mengekstrak data entitas kontak dan profil institusi pendidikan dari teks secara otomatis."""
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        phone_pattern = r'(\+62|0)[0-9\s-]{9,14}'
        
        email_match = re.search(email_pattern, cv_text)
        phone_match = re.search(phone_pattern, cv_text)
        
        univ_pattern = r'(universitas|university|institut|politeknik|academy)\s+[\w\s]+'
        univ_match = re.search(univ_pattern, cv_text, re.IGNORECASE)
        
        major_pattern = r'(jurusan|program studi|prodi|major|departemen|sains data|informatika|teknologi)\s+[\w\s]+'
        major_match = re.search(major_pattern, cv_text, re.IGNORECASE)
        
        return {
            "email": email_match.group(0) if email_match else "Tidak terdeteksi",
            "phone": phone_match.group(0).strip() if phone_match else "Tidak terdeteksi",
            "institution": univ_match.group(0).strip() if univ_match else "Tidak terdeteksi",
            "major": major_match.group(0).strip() if major_match else "Tidak terdeteksi"
        }

    # =========================================================================
    # [FITUR 4] PREDICTIVE INTERVIEW PREP (SIMULATOR PERTANYAAN WAWANCARA)
    # =========================================================================
    def generate_predictive_interview_prep(self, missing_skills: List[str]) -> List[str]:
        """Menghasilkan simulasi pertanyaan teknis spesifik berdasarkan keahlian yang belum terpenuhi (gap)."""
        questions = []
        
        # Bank pertanyaan teknis industri berdasarkan kecocokan kata kunci keahlian
        interview_question_bank = {
            "machine learning": "Bisakah Anda menjelaskan alur lengkap rekayasa fitur (feature engineering) sebelum melatih model Machine Learning?",
            "deep learning": "Bagaimana Anda mengatasi kendala vanishing gradient atau overfitting saat melatih model deep neural network?",
            "keras": "Kapan Anda sebaiknya menggunakan Sequential API dibandingkan Functional API pada pengembangan model di Keras?",
            "python": "Bagaimana cara Anda mengoptimalkan struktur data atau pemrosesan paralel agar eksekusi skrip Python berjalan efisien?",
            "sql": "Jelaskan konsep pengindeksan (indexing) database dan dampaknya terhadap performa query pemrosesan data berskala besar.",
            "go": "Bagaimana pemanfaatan mekanisme Goroutines dan Channels untuk menangani masalah konkurensi di bahasa Go?",
            "r studio": "Bagaimana Anda melakukan penanganan nilai hilang (missing values) dan uji validitas statistika di R Studio?",
            "figma": "Bagaimana langkah-langkah nyata Anda dalam memvalidasi sebuah komponen desain antarmuka melalui tahapan Usability Testing?"
        }
        
        count = 0
        for skill in missing_skills:
            skill_lower = skill.lower()
            if skill_lower in interview_question_bank:
                questions.append(interview_question_bank[skill_lower])
                count += 1
            if count >= 3:  
                break
                
        # Jika missing skills tidak masuk bank data teknis, berikan pertanyaan evaluasi berbasis pengalaman/perilaku
        if len(questions) < 3:
            generic_questions = [
                "Bagaimana Anda memprioritaskan penyelesaian tugas teknis ketika menghadapi tenggat waktu (deadline) proyek tim yang sangat ketat?",
                "Ceritakan pengalaman Anda saat harus memecahkan kendala eror sistem yang belum pernah Anda temui solusinya di dokumentasi resmi.",
                "Bagaimana cara Anda mengomunikasikan hasil analisis teknis yang rumit kepada pemangku kepentingan (stakeholders) non-teknis?"
            ]
            questions.extend(generic_questions[:3 - len(questions)])
            
        return questions

    def extract_soft_skills(self, cv_text: str) -> List[Dict]:
        text = cv_text.lower()
        soft_skills_lexicon = {
            "Leadership": ["lead", "manage", "mentor", "memimpin", "mengelola", "head", "manager", "ketua", "koordinator"],
            "Teamwork": ["collaborate", "team", "support", "kolaborasi", "tim", "bersama", "assist", "panitia", "anggota"],
            "Communication": ["present", "communicate", "negotiate", "komunikasi", "presentasi", "negosiasi", "speaker", "humas"],
            "Problem Solving": ["solve", "analyze", "resolve", "troubleshoot", "analisis", "menyelesaikan", "solusi", "riset"],
            "Adaptability": ["adapt", "flexible", "agile", "adaptasi", "fleksibel", "dinamis", "cepat belajar", "mandiri"]
        }
        
        detected_soft_skills = []
        for skill_name, keywords in soft_skills_lexicon.items():
            count = sum(len(re.findall(rf'\b{kw}\b', text)) for kw in keywords)
            if count > 0:
                level = "Tinggi" if count >= 3 else "Menengah" if count == 2 else "Dasar"
                detected_soft_skills.append({
                    "name": skill_name,
                    "level": level,
                    "count": count
                })
                
        return sorted(detected_soft_skills, key=lambda x: x['count'], reverse=True)

    def evaluate_cv_quality(self, cv_text: str) -> Dict:
        text = cv_text.lower()
        metrics = re.findall(r'(\d+%)|(rp\s*\d+)|(\$\d+)|(\d+\+)|(\d+\s*(juta|ribu|k|rb))', text)
        metric_score = min(len(metrics) * 15, 40) 
        
        action_verbs = ['led', 'memimpin', 'developed', 'mengembangkan', 'created', 'membuat', 
                        'managed', 'mengelola', 'increased', 'meningkatkan', 'achieved', 'mencapai',
                        'designed', 'merancang', 'optimized', 'menganalisis', 'menyelenggarakan']
        
        verb_count = sum(1 for verb in action_verbs if verb in text)
        verb_score = min(verb_count * 5, 40) 
        
        structure_score = 0
        if any(w in text for w in ['education', 'pendidikan', 'university', 'universitas', 'sekolah', 'institut']):
            structure_score += 10
        if any(w in text for w in ['experience', 'pengalaman', 'work', 'kerja', 'project', 'proyek', 'kepanitiaan']):
            structure_score += 10
            
        total_score = metric_score + verb_score + structure_score
        
        advice = []
        if metric_score < 20:
            advice.append("Tambahkan angka metrik pencapaian (misal: 'berhasil meningkatkan akurasi model sebesar 12%').")
        if verb_score < 15:
            advice.append("Gunakan kata kerja aksi aktif di awal poin kegiatan Anda (seperti 'Merancang', 'Menginisiasi').")
        if structure_score < 20:
            advice.append("Pastikan susunan tata letak memisahkan riwayat pendidikan dan pengalaman organisasi/proyek secara jelas.")
            
        return {
            "score": total_score,
            "metric_count": len(metrics),
            "verb_count": verb_count,
            "advice": advice if advice else ["Struktur dan bobot kualitas deskripsi CV Anda sudah optimal."]
        }

    def generate_advice(self, role: str, missing: List[str], match_score: float) -> str:
        if match_score >= 80 and len(missing) <= 1:
            return f"Kecocokan semantik tinggi! Profil Anda sangat direkomendasikan untuk posisi {role}."
        if len(missing) == 0:
            return "Luar biasa! Seluruh kata kunci keahlian utama terdeteksi dengan kecocokan penuh."
            
        advice = f"Untuk peran {role}, Anda disarankan memperkuat kompetensi: {', '.join(missing[:3])}. "
        if match_score < 40:
            advice += "Sistem mendeteksi gap relevansi yang cukup besar. Pertimbangkan membangun portofolio proyek baru."
        else:
            advice += "Gunakan istilah kompetensi teknis secara eksplisit agar mudah terbaca oleh mesin parsing."
        return advice

    # =========================================================================
    # PERUBAHAN PADA FUNGSI UTAMA PIPELINE (INTEGRASI SEMUA FITUR)
    # =========================================================================
    def analyze_cv(self, pdf_path: str, top_n: int = 3) -> Dict:
        cv_text = self.extract_text_from_pdf(pdf_path)
        if not cv_text:
            return {"error": "Sistem gagal memproses ekstraksi teks dokumen PDF."}

        # Menjalankan Fitur Ekstraksi Data Profil Otomatis (Auto-Fill Form)
        parsed_profile = self.parse_basic_info(cv_text)
        
        # Menjalankan Analisis Kualitas & Gaya Kalimat
        quality_report = self.evaluate_cv_quality(cv_text)
        tailor_suggestions = self.generate_resume_tailor_suggestions(cv_text)
        soft_skills_report = self.extract_soft_skills(cv_text)

        # Proses Vektor Semantik (LSA & SBERT Hybrid)
        cv_tfidf = self.vectorizer.transform([cv_text])
        cv_semantic = self.lsa_model.transform(cv_tfidf)
        
        # 1. Hitung similarity menggunakan TF-IDF (Leksikal)
        tfidf_sim = cosine_similarity(cv_tfidf, self.job_tfidf)[0]
        
        # 2. Hubungi SBERT API jika memungkinkan
        sbert_sim = None
        if self.job_sbert is not None:
            cv_sbert = self._get_sbert_embeddings([cv_text])
            if cv_sbert is not None:
                sbert_sim = cosine_similarity(cv_sbert, self.job_sbert)[0]
        
        # 3. Hitung skor akhir hibrida atau fallback ke LSA
        if sbert_sim is not None:
            # Formula hibrida: 40% TF-IDF (Exact Match) + 60% SBERT (Contextual Semantics)
            similarities = 0.4 * tfidf_sim + 0.6 * sbert_sim
            is_hybrid = True
        else:
            # Fallback ke LSA klasik jika SBERT offline
            similarities = cosine_similarity(cv_semantic, self.job_semantic)[0]
            is_hybrid = False
        
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        matches = []
        all_missing_skills = []
        primary_category = "Data & AI"  # Default fallback
        
        for idx in top_indices:
            raw_score = float(similarities[idx])
            if is_hybrid:
                score = max(0.0, min(100.0, (raw_score * 100) + 10))
            else:
                score = max(0.0, min(100.0, (raw_score * 100) + 15))
                
            row = self.jobs_df.iloc[idx]
            primary_category = str(row['category'])
            req_skills = [s.strip() for s in str(row['skills']).split(',')]
            
            matched = []
            missing = []
            cv_lower = cv_text.lower()
            
            radar_labels = []
            radar_data = []
            
            for skill in req_skills:
                if len(radar_labels) < 6:
                    radar_labels.append(skill[:15]) 
                
                is_match = bool(re.search(r'\b' + re.escape(skill.lower()) + r'\b', cv_lower))
                
                if is_match:
                    matched.append(skill)
                    if len(radar_labels) <= 6: radar_data.append(10)
                else:
                    missing.append(skill)
                    all_missing_skills.append(skill)
                    if len(radar_labels) <= 6: radar_data.append(2)
            
            while len(radar_labels) < 3:
                radar_labels.append("Umum")
                radar_data.append(5)
            
            advice = self.generate_advice(str(row['role']), missing, score)
            
            matches.append(JobMatch(
                job_title=str(row['role']),
                category=primary_category,
                similarity_score=round(score, 1),
                matched_skills=matched,
                missing_skills=missing,
                description=str(row['desc']),
                required_skills_raw=str(row['skills']),
                improvement_advice=advice,
                radar_labels=radar_labels,
                radar_data=radar_data
            ))
            
        # Menjalankan Fitur Simulator Pertanyaan Wawancara Berdasarkan Gap
        interview_prep = self.generate_predictive_interview_prep(all_missing_skills)
        
        # Menjalankan Fitur Statistik Pasar Kerja Berdasarkan Kategori Utama yang Cocok
        market_insights = self.get_market_insights(primary_category)
        
        # Mengembalikan semua payload data untuk di-render ke antarmuka aplikasi
        return {
            "parsed_profile": parsed_profile,
            "quality": quality_report,
            "tailor_suggestions": tailor_suggestions,
            "soft_skills": soft_skills_report,
            "matches": matches,
            "interview_prep": interview_prep,
            "market_insights": market_insights,
            "keywords": self.extract_keywords_from_cv(cv_text)
        }

    def extract_keywords_from_cv(self, text: str) -> List[str]:
        vec = TfidfVectorizer(stop_words=self.vectorizer.stop_words, max_features=10)
        try:
            vec.fit([text])
            return list(vec.get_feature_names_out())
        except:
            return []