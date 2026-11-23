
# Each question lists EVERY invoice that correctly answers it.
# A retrieval counts as a hit if any of the top-k results is in that list.
# (Many invoices share a customer or a product, so a single label would
# mark correct answers as misses.)

evaluation_data = [
    {
        "question": "Find the invoice with Order ID 11062",
        "expected_invoices": ["invoice_11062.pdf"]
    },
    {
        "question": "Which customer placed Order ID 11062?",
        "expected_invoices": ["invoice_11062.pdf"]
    },
    {
        "question": "Find the invoice for customer DRACD",
        "expected_invoices": [
            "invoice_10363.pdf", "invoice_10391.pdf", "invoice_10797.pdf",
            "invoice_10825.pdf", "invoice_11036.pdf", "invoice_11067.pdf"
        ]
    },
    {
        "question": "Find the invoice containing Pâté chinois",
        "expected_invoices": [
            "invoice_10254.pdf", "invoice_10280.pdf", "invoice_10342.pdf",
            "invoice_10356.pdf", "invoice_10389.pdf", "invoice_10398.pdf",
            "invoice_10425.pdf", "invoice_10451.pdf", "invoice_10461.pdf",
            "invoice_10476.pdf", "invoice_10485.pdf", "invoice_10529.pdf",
            "invoice_10559.pdf", "invoice_10564.pdf", "invoice_10606.pdf",
            "invoice_10615.pdf", "invoice_10630.pdf", "invoice_10668.pdf",
            "invoice_10707.pdf", "invoice_10725.pdf", "invoice_10728.pdf",
            "invoice_10802.pdf", "invoice_10872.pdf", "invoice_10902.pdf",
            "invoice_10930.pdf", "invoice_10948.pdf", "invoice_10990.pdf",
            "invoice_11001.pdf", "invoice_11002.pdf", "invoice_11028.pdf",
            "invoice_11056.pdf", "invoice_11064.pdf", "invoice_11077.pdf"
        ]
    },
    {
        "question": "Which invoice contains Escargots de Bourgogne?",
        "expected_invoices": [
            "invoice_10314.pdf", "invoice_10327.pdf", "invoice_10373.pdf",
            "invoice_10374.pdf", "invoice_10537.pdf", "invoice_10594.pdf",
            "invoice_10636.pdf", "invoice_10661.pdf", "invoice_10675.pdf",
            "invoice_10697.pdf", "invoice_10714.pdf", "invoice_10839.pdf",
            "invoice_10845.pdf", "invoice_10863.pdf", "invoice_10904.pdf",
            "invoice_10913.pdf", "invoice_10930.pdf", "invoice_11011.pdf"
        ]
    },
    {
        "question": "Find the invoice containing Gudbrandsdalsost",
        "expected_invoices": [
            "invoice_10296.pdf", "invoice_10308.pdf", "invoice_10311.pdf",
            "invoice_10350.pdf", "invoice_10356.pdf", "invoice_10364.pdf",
            "invoice_10392.pdf", "invoice_10395.pdf", "invoice_10407.pdf",
            "invoice_10419.pdf", "invoice_10493.pdf", "invoice_10529.pdf",
            "invoice_10552.pdf", "invoice_10583.pdf", "invoice_10593.pdf",
            "invoice_10595.pdf", "invoice_10640.pdf", "invoice_10667.pdf",
            "invoice_10693.pdf", "invoice_10717.pdf", "invoice_10746.pdf",
            "invoice_10747.pdf", "invoice_10752.pdf", "invoice_10755.pdf",
            "invoice_10756.pdf", "invoice_10796.pdf", "invoice_10894.pdf",
            "invoice_10962.pdf", "invoice_10964.pdf", "invoice_11022.pdf",
            "invoice_11033.pdf"
        ]
    },
    {
        "question": "Which invoices contain Mishi Kobe Niku?",
        "expected_invoices": [
            "invoice_10420.pdf", "invoice_10515.pdf", "invoice_10687.pdf",
            "invoice_10693.pdf", "invoice_10848.pdf"
        ]
    }
]
