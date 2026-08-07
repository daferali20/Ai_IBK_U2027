Ai_IBK_U2027/
├── app.py                          # التطبيق الرئيسي
├── requirements.txt                # المتطلبات
├── runtime.txt                     # إصدار Python
├── .gitignore                      # تجاهل الملفات
├── README.md                       # توثيق المشروع
│
├── core/                           # النواة الأساسية
│   ├── __init__.py
│   ├── config.py                   # الإعدادات
│   └── constants.py                # الثوابت
│
├── models/                         # نماذج الذكاء الاصطناعي
│   ├── __init__.py
│   ├── base_model.py               # النموذج الأساسي
│   └── model_utils.py              # أدوات النماذج
│
├── data/                           # معالجة البيانات
│   ├── __init__.py
│   ├── fetcher.py                  # جلب البيانات
│   └── indicators.py               # المؤشرات الفنية
│
├── brokers/                        # الوسطاء
│   ├── __init__.py
│   ├── base_broker.py              # الواجهة الأساسية
│   └── ibkr_broker.py              # IBKR
│
├── strategies/                     # الاستراتيجيات
│   ├── __init__.py
│   ├── ml_strategy.py              # التعلم الآلي
│   └── hybrid_strategy.py          # الهجينة
│
├── ui/                             # واجهة المستخدم
│   ├── __init__.py
│   ├── sidebar.py                  # الشريط الجانبي
│   ├── charts.py                   # الرسوم البيانية
│   └── components.py               # مكونات الواجهة
│
└── utils/                          # أدوات مساعدة
    ├── __init__.py
    ├── helpers.py                  # دوال مساعدة
    └── logger.py                   # تسجيل الأخطاء
