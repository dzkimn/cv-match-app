import pandas as pd

jobs_data = [
    # Data & AI
    {"role": "Data Scientist", "category": "Data & AI", "desc": "Menganalisis data besar untuk menemukan pola dan membangun model prediktif.", "skills": "Python, SQL, Machine Learning, Deep Learning, Statistics, Pandas, Scikit-learn, TensorFlow"},
    {"role": "Data Analyst", "category": "Data & AI", "desc": "Menerjemahkan data menjadi laporan bisnis untuk pengambilan keputusan.", "skills": "SQL, Excel, Tableau, Power BI, Python, Data Visualization, Analytical Skills"},
    {"role": "Data Engineer", "category": "Data & AI", "desc": "Membangun arsitektur data dan pipeline ETL yang skalabel.", "skills": "Python, SQL, Apache Spark, Hadoop, AWS, GCP, ETL, Data Warehousing"},
    {"role": "Machine Learning Engineer", "category": "Data & AI", "desc": "Mendeploy dan mengoptimalkan model AI ke dalam sistem produksi.", "skills": "Python, C++, PyTorch, MLOps, Docker, Kubernetes, CI/CD"},
    {"role": "Business Intelligence Analyst", "category": "Data & AI", "desc": "Menganalisis metrik bisnis menggunakan dashboard interaktif.", "skills": "SQL, Power BI, Tableau, Business Strategy, Data Modeling"},
    {"role": "AI Prompt Engineer", "category": "Data & AI", "desc": "Merancang dan mengoptimalkan prompt untuk Large Language Models.", "skills": "NLP, Prompt Engineering, OpenAI API, Python, Creative Writing"},
    {"role": "Computer Vision Engineer", "category": "Data & AI", "desc": "Membangun sistem AI untuk pengenalan gambar dan video.", "skills": "Python, OpenCV, PyTorch, CNN, Image Processing"},
    {"role": "NLP Engineer", "category": "Data & AI", "desc": "Mengembangkan sistem AI pemrosesan bahasa alami.", "skills": "Python, NLP, Hugging Face, Transformers, NLTK, Spacy"},
    {"role": "Deep Learning Engineer", "category": "Data & AI", "desc": "Membangun dan melatih model deep learning tingkat lanjut untuk pengenalan pola.", "skills": "Python, PyTorch, TensorFlow, Keras, CNN, RNN, Transformers, GPU"},
    {"role": "Data Architect", "category": "Data & AI", "desc": "Merancang dan mengelola arsitektur penyimpanan data terdistribusi dan sistem database.", "skills": "Data Modeling, SQL, NoSQL, Hadoop, Spark, Snowflake, AWS, Azure"},
    {"role": "Data Quality Specialist", "category": "Data & AI", "desc": "Memastikan kebersihan, konsistensi, dan integritas data dalam data warehouse.", "skills": "SQL, Python, Data Quality, ETL, Excel, Reporting"},
    
    # Software Engineering
    {"role": "Frontend Developer", "category": "Software Engineering", "desc": "Membangun antarmuka web (Client-side) yang responsif dan interaktif.", "skills": "HTML, CSS, JavaScript, React.js, Vue.js, Tailwind, Responsive Design"},
    {"role": "Backend Developer", "category": "Software Engineering", "desc": "Membangun logika server, database, dan RESTful API.", "skills": "Node.js, Go, Python, Java, PostgreSQL, MongoDB, REST API, Microservices"},
    {"role": "Fullstack Developer", "category": "Software Engineering", "desc": "Menguasai pengembangan sisi klien maupun sisi server.", "skills": "JavaScript, React.js, Node.js, Express, MongoDB, SQL, Git"},
    {"role": "Android Developer", "category": "Software Engineering", "desc": "Mengembangkan aplikasi mobile khusus untuk sistem operasi Android.", "skills": "Kotlin, Java, Android Studio, XML, REST API, SQLite"},
    {"role": "iOS Developer", "category": "Software Engineering", "desc": "Mengembangkan aplikasi mobile ekosistem Apple.", "skills": "Swift, Objective-C, Xcode, iOS SDK, Core Data"},
    {"role": "Flutter Developer", "category": "Software Engineering", "desc": "Membangun aplikasi mobile cross-platform menggunakan Flutter.", "skills": "Flutter, Dart, Firebase, REST API, State Management"},
    {"role": "QA Engineer (Tester)", "category": "Software Engineering", "desc": "Memastikan kualitas perangkat lunak dengan automated & manual testing.", "skills": "Selenium, Cypress, Jest, Manual Testing, Automation Testing, QA"},
    {"role": "QA Automation Engineer", "category": "Software Engineering", "desc": "Menulis skrip uji coba otomatis tingkat lanjut untuk aplikasi web dan mobile.", "skills": "Selenium, Appium, Python, JavaScript, Jenkins, CI/CD, Test Automation"},
    {"role": "Mobile Tech Lead", "category": "Software Engineering", "desc": "Memimpin pengembangan teknis tim pengembang aplikasi mobile (iOS & Android).", "skills": "Swift, Kotlin, Flutter, Mobile Architecture, Code Review, Team Leadership"},
    {"role": "Game Developer", "category": "Software Engineering", "desc": "Membangun logika permainan, karakter, dan lingkungan interaktif.", "skills": "Unity, C#, Unreal Engine, C++, Game Design, 3D Math"},
    {"role": "Blockchain Developer", "category": "Software Engineering", "desc": "Membangun sistem terdesentralisasi dan smart contracts.", "skills": "Solidity, Web3.js, Ethereum, Cryptography, Blockchain Architecture"},
    
    # Infrastructure & Operations
    {"role": "DevOps Engineer", "category": "Infrastructure", "desc": "Menjembatani pengembangan dan operasi IT, fokus pada CI/CD.", "skills": "Linux, Docker, Kubernetes, AWS, Jenkins, Terraform, CI/CD"},
    {"role": "Cloud Architect", "category": "Infrastructure", "desc": "Merancang arsitektur komputasi awan untuk perusahaan.", "skills": "AWS, Azure, GCP, Cloud Security, System Architecture, Networking"},
    {"role": "Database Administrator", "category": "Infrastructure", "desc": "Memelihara dan mengamankan sistem database relasional/non-relasional.", "skills": "SQL, PostgreSQL, MySQL, Oracle, Database Tuning, Backup & Recovery"},
    {"role": "Network Engineer", "category": "Infrastructure", "desc": "Mengonfigurasi dan memelihara infrastruktur jaringan kabel dan nirkabel perusahaan.", "skills": "Cisco, CCNA, Routing & Switching, TCP/IP, VPN, Firewalls, DNS"},
    {"role": "Site Reliability Engineer (SRE)", "category": "Infrastructure", "desc": "Menjaga keandalan, ketersediaan, dan skalabilitas layanan sistem produksi berskala besar.", "skills": "Python, Linux, Prometheus, Grafana, Docker, Kubernetes, CI/CD, Terraform"},
    {"role": "Solutions Architect", "category": "Infrastructure", "desc": "Merancang arsitektur perangkat lunak komprehensif yang memecahkan masalah bisnis spesifik.", "skills": "System Design, Microservices, Cloud Platforms, Enterprise Architecture, API Design"},
    {"role": "IT Support Specialist", "category": "IT Operations", "desc": "Memberikan bantuan teknis untuk hardware dan software perusahaan.", "skills": "Troubleshooting, Windows Server, Networking, Helpdesk, Hardware"},
    {"role": "System Administrator", "category": "IT Operations", "desc": "Mengelola dan memelihara server dan jaringan komputer perusahaan.", "skills": "Linux, Windows Server, Active Directory, Bash Scripting, Networking"},
    {"role": "Systems Engineer", "category": "IT Operations", "desc": "Mengintegrasikan berbagai sistem IT, hardware, dan software agar bekerja secara selaras.", "skills": "Linux, Windows Server, Virtualization, VMware, Bash Scripting, Active Directory"},
    {"role": "System Integration Engineer", "category": "IT Operations", "desc": "Menyambungkan sistem perangkat lunak warisan dengan API modern.", "skills": "REST API, Python, SOAP, System Integration, Enterprise Architecture"},

    # Security
    {"role": "Cyber Security Analyst", "category": "Security", "desc": "Memantau jaringan IT untuk mendeteksi ancaman dan serangan siber.", "skills": "Network Security, SIEM, Firewalls, Penetration Testing, Wireshark"},
    {"role": "Penetration Tester (Ethical Hacker)", "category": "Security", "desc": "Mencari celah keamanan sistem dengan melakukan simulasi serangan.", "skills": "Ethical Hacking, Metasploit, Kali Linux, Web Security, OWASP"},
    {"role": "Security Engineer", "category": "Security", "desc": "Mengimplementasikan solusi keamanan sistem, jaringan, dan perangkat lunak di internal perusahaan.", "skills": "Cryptography, Identity & Access Management (IAM), Firewalls, Linux, AWS Security"},
    {"role": "IT Auditor", "category": "Security", "desc": "Mengaudit kepatuhan sistem keamanan IT perusahaan terhadap regulasi standar industri.", "skills": "IT Audit, COBIT, ISO 27001, Risk Assessment, Compliance, Security Control"},

    # Design & Product
    {"role": "UI/UX Designer", "category": "Design", "desc": "Merancang pengalaman dan antarmuka pengguna aplikasi.", "skills": "Figma, Adobe XD, Wireframing, Prototyping, User Research, UI Design"},
    {"role": "UX Researcher", "category": "Design", "desc": "Melakukan riset mendalam terhadap perilaku dan kebutuhan pengguna.", "skills": "User Research, Usability Testing, Interviewing, Data Analysis, Psychology"},
    {"role": "Product Designer", "category": "Design", "desc": "Menggabungkan desain UI/UX dengan pemahaman bisnis untuk menciptakan produk digital utuh.", "skills": "Figma, Adobe XD, Product Strategy, Interaction Design, Prototyping, Design Systems"},
    {"role": "Graphic Designer", "category": "Design", "desc": "Membuat aset visual untuk branding dan pemasaran digital.", "skills": "Adobe Illustrator, Photoshop, CorelDraw, Visual Design, Typography"},
    {"role": "Product Manager", "category": "Product", "desc": "Mengelola siklus hidup produk dari ide hingga peluncuran.", "skills": "Product Strategy, Agile, Scrum, Leadership, Market Research, JIRA"},
    {"role": "Project Manager", "category": "Product", "desc": "Merencanakan dan memastikan proyek selesai tepat waktu.", "skills": "Project Management, Agile, Scrum, Trello, Asana, Communication"},
    {"role": "Scrum Master", "category": "Product", "desc": "Memfasilitasi tim agile agar bekerja efektif sesuai framework Scrum.", "skills": "Scrum, Agile Methodology, Leadership, Sprint Planning, Coaching"},
    {"role": "Product Owner", "category": "Product", "desc": "Mendefinisikan visi produk dan mengelola backlog tim pengembang secara agile.", "skills": "Agile, Scrum, Product Backlog, User Stories, Product Strategy, JIRA"},

    # Marketing & Business
    {"role": "Digital Marketing Specialist", "category": "Marketing", "desc": "Merencanakan dan mengeksekusi kampanye pemasaran digital.", "skills": "SEO, SEM, Social Media Marketing, Google Ads, Content Strategy"},
    {"role": "SEO Specialist", "category": "Marketing", "desc": "Mengoptimalkan website agar mendapat peringkat tinggi di mesin pencari.", "skills": "SEO, Google Analytics, Ahrefs, Keyword Research, Content Writing"},
    {"role": "Social Media Manager", "category": "Marketing", "desc": "Mengelola akun media sosial dan membangun interaksi dengan audiens.", "skills": "Social Media Strategy, Copywriting, Content Creation, Hootsuite, Analytics"},
    {"role": "Content Writer", "category": "Marketing", "desc": "Menulis konten menarik untuk blog, artikel, dan media sosial.", "skills": "Copywriting, SEO Writing, Storytelling, Research, Editing"},
    {"role": "Copywriter", "category": "Marketing", "desc": "Menulis teks persuasif untuk tujuan periklanan dan penjualan.", "skills": "Copywriting, Persuasion, Advertising, Creative Writing, Marketing"},
    {"role": "Brand Manager", "category": "Marketing", "desc": "Mengelola identitas, persepsi, dan kampanye branding dari suatu produk atau merek dagang.", "skills": "Brand Strategy, Market Analysis, Marketing Campaigns, Public Relations, Leadership"},
    {"role": "Marketing Analyst", "category": "Marketing", "desc": "Mengukur dan mengevaluasi efektivitas ROI dari berbagai saluran kampanye pemasaran.", "skills": "Google Analytics, SQL, Excel, Tableau, Marketing Analytics, A/B Testing"},
    {"role": "Public Relations Specialist", "category": "Marketing", "desc": "Membangun dan memelihara hubungan positif antara perusahaan dengan media massa dan publik.", "skills": "Public Relations, Copywriting, Media Relations, Communication, Crisis Management"},
    {"role": "Business Analyst", "category": "Business", "desc": "Menganalisis proses bisnis dan merekomendasikan solusi teknologi.", "skills": "Business Analysis, BPMN, Requirement Gathering, SQL, Communication"},
    {"role": "Sales Executive", "category": "Business", "desc": "Mencari prospek klien baru dan menutup kesepakatan penjualan.", "skills": "B2B Sales, Negotiation, CRM, Communication, Lead Generation"},
    {"role": "Key Account Manager", "category": "Business", "desc": "Membangun hubungan jangka panjang dengan klien VIP perusahaan.", "skills": "Account Management, CRM, Negotiation, Relationship Building"},
    {"role": "Business Development Manager", "category": "Business", "desc": "Mencari peluang kemitraan bisnis baru dan memperluas jangkauan pasar perusahaan.", "skills": "Business Development, Negotiation, Sales Strategy, Partnerships, Communication"},
    {"role": "ERP Specialist", "category": "Business", "desc": "Mengonfigurasi dan memelihara sistem ERP perusahaan seperti SAP atau Oracle.", "skills": "SAP, ERP Systems, Business Process, SQL, Database Management"},
    
    # Finance & HR
    {"role": "Financial Analyst", "category": "Finance", "desc": "Menganalisis data keuangan untuk memprediksi kondisi ekonomi perusahaan.", "skills": "Financial Modeling, Excel, Forecasting, Accounting, Valuation"},
    {"role": "Accountant", "category": "Finance", "desc": "Mencatat dan mengelola laporan keuangan perusahaan.", "skills": "Accounting, Tax, Bookkeeping, SAP, Excel, Audit"},
    {"role": "HR Generalist", "category": "Human Resources", "desc": "Mengelola rekrutmen, payroll, dan hubungan karyawan.", "skills": "Recruitment, Payroll, HRIS, Employee Relations, Labor Law"},
    {"role": "Technical Recruiter", "category": "Human Resources", "desc": "Mencari dan merekrut talenta spesifik di bidang IT.", "skills": "Sourcing, LinkedIn Recruiter, Interviewing, Tech Savy, Networking"},
    {"role": "HR Manager", "category": "Human Resources", "desc": "Memimpin departemen SDM, merumuskan kebijakan ketenagakerjaan, dan mengelola retensi karyawan.", "skills": "Human Resources, Leadership, Employee Relations, Labor Law, Talent Management"},
    {"role": "Talent Acquisition Specialist", "category": "Human Resources", "desc": "Fokus pada pencarian pasif, wawancara, dan akuisisi talenta berbakat secara proaktif.", "skills": "Recruitment, Sourcing, LinkedIn Recruiter, Candidate Experience, Interviewing"},
    
    # Engineering & Creative
    {"role": "Mechanical Engineer", "category": "Engineering", "desc": "Merancang sistem mekanis dan mesin.", "skills": "AutoCAD, SolidWorks, Mechanical Design, Thermodynamics, Manufacturing"},
    {"role": "Electrical Engineer", "category": "Engineering", "desc": "Mendesain sistem kelistrikan dan sirkuit elektronik.", "skills": "Electrical Design, MATLAB, Circuit Design, Power Systems, PCB Design"},
    {"role": "Civil Engineer", "category": "Engineering", "desc": "Merancang dan mengawasi proyek infrastruktur dan konstruksi.", "skills": "AutoCAD, SAP2000, Structural Analysis, Construction Management"},
    {"role": "Embedded Systems Engineer", "category": "Engineering", "desc": "Mengembangkan perangkat lunak level rendah (firmware) untuk perangkat keras elektronik.", "skills": "C, C++, Microcontrollers, RTOS, Firmware, IoT, Hardware Debugging"},
    {"role": "Video Editor", "category": "Creative", "desc": "Mengedit dan menggabungkan potongan video menjadi karya utuh.", "skills": "Premiere Pro, After Effects, Final Cut Pro, Color Grading, Motion Graphics"},
    {"role": "3D Artist", "category": "Creative", "desc": "Membuat model dan aset 3D untuk game, film, atau media visual interaktif.", "skills": "Blender, Maya, ZBrush, 3D Modeling, Texturing, Lighting"},
    {"role": "Animator", "category": "Creative", "desc": "Menghidupkan karakter dan objek visual menggunakan teknik animasi 2D atau 3D.", "skills": "Adobe After Effects, Blender, Maya, 2D Animation, 3D Animation, Storyboarding"},
    
    # Operations
    {"role": "Customer Success Specialist", "category": "Operations", "desc": "Membantu pelanggan mencapai hasil maksimal dengan produk perusahaan.", "skills": "Customer Service, Communication, Problem Solving, CRM, Empathy"},
    {"role": "Supply Chain Analyst", "category": "Operations", "desc": "Menganalisis dan mengoptimalkan logistik dan rantai pasokan.", "skills": "Supply Chain Management, Logistics, Excel, Data Analysis, Inventory Management"},
    {"role": "Operations Manager", "category": "Operations", "desc": "Mengawasi jalannya operasional harian perusahaan agar berjalan efisien dan produktif.", "skills": "Operations Management, Budgeting, Leadership, Process Improvement, Business Strategy"}
]

df = pd.DataFrame(jobs_data)
df.to_csv("job_roles_dataset.csv", index=False)
print(f"Dataset generated with {len(df)} roles!")
