"""Türkçe hukuk metinleri (demo amaçlı mini-corpus)."""

DEMO_DOCS = [
    {
        "id": "anayasa_1982",
        "text": (
            "Türkiye Cumhuriyeti Anayasası 1982 yılında halkoylaması ile kabul edildi. "
            "Anayasa, devletin temel yapısını, organların görev ve yetkilerini düzenler. "
            "1982 Anayasası 177 madde ve geçici maddelerden oluşur. "
            "Cumhurbaşkanı, TBMM ve yargı erkleri ayrı ayrı düzenlenmiştir."
        ),
        "meta": {"source": "MEB Vatandaşlık Bilgisi"},
    },
    {
        "id": "is_kanunu_4857",
        "text": (
            "4857 sayılı İş Kanunu işçi-işveren ilişkilerini düzenler. "
            "Belirsiz süreli iş sözleşmesinin feshi için bildirim süreleri çalışma süresine bağlıdır: "
            "6 aydan az çalışana 2 hafta, 6 ay-1.5 yıl arası 4 hafta, "
            "1.5-3 yıl arası 6 hafta, 3 yıldan fazla 8 hafta bildirim süresi uygulanır. "
            "İhbar tazminatı bu sürelere göre hesaplanır."
        ),
        "meta": {"source": "4857 sayılı İş Kanunu"},
    },
    {
        "id": "kvkk_6698",
        "text": (
            "6698 sayılı Kişisel Verilerin Korunması Kanunu (KVKK) 7 Nisan 2016'da yürürlüğe girdi. "
            "Kişisel verilerin işlenmesinde açık rıza esastır. "
            "VERBİS (Veri Sorumluları Sicili) kayıt yükümlülüğü belirli kriterlere göre uygulanır. "
            "Açık rıza alınmadan veri işlenmesi sadece kanunda belirtilen istisnai durumlarda mümkündür."
        ),
        "meta": {"source": "KVKK"},
    },
    {
        "id": "tck_5237",
        "text": (
            "5237 sayılı Türk Ceza Kanunu 2005 yılında yürürlüğe girdi. "
            "Suç ve cezaların kanuniliği ilkesi esas alınır. "
            "Kasten adam öldürmenin cezası müebbet hapis cezasıdır, "
            "ağırlaştırılmış müebbet hapis cezası belirli durumlarda uygulanır."
        ),
        "meta": {"source": "5237 sayılı TCK"},
    },
    {
        "id": "ttk_6102",
        "text": (
            "6102 sayılı Türk Ticaret Kanunu 2012 yılında yürürlüğe girdi. "
            "Anonim şirketlerin asgari sermayesi 50.000 TL, "
            "limited şirketlerin asgari sermayesi 10.000 TL olarak belirlenmiştir. "
            "Şirketler tescil ve ilanla tüzel kişilik kazanır."
        ),
        "meta": {"source": "6102 sayılı TTK"},
    },
    {
        "id": "borclar_6098",
        "text": (
            "6098 sayılı Türk Borçlar Kanunu 2012 yılında yürürlüğe girdi. "
            "Sözleşmeler kural olarak şekle bağlı değildir, ancak bazı sözleşmeler için "
            "kanun yazılı şekil veya resmi şekil şart koşmuştur. "
            "Taşınmaz satışı resmi şekilde, kefalet ise yazılı olarak yapılmalıdır."
        ),
        "meta": {"source": "6098 sayılı TBK"},
    },
]
