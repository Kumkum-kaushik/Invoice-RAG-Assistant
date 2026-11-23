
# RAGAS context-evaluation dataset.
#
# Every fact below was checked against data/processed/extracted_invoices.json.
#
# References are written as SEPARATE short statements, one fact each.
# ContextRecall splits the reference into statements and scores the share
# that the retrieved contexts support; ContextPrecision asks, per context,
# whether it helps arrive at the reference. A single sentence listing six
# invoices is all-or-nothing for both (the original cause of the 0.0 scores),
# so multi-invoice answers are one statement per invoice.
#
# Note: with top_k=3, a question with 5-6 relevant invoices can reach at most
# about 3/5-3/6 context recall. That is a real recall limit, not an error.

ragas_evaluation_data = [

    # 1. Exact invoice lookup
    {
        "question": "Find the invoice with Order ID 11062",
        "type": "exact_invoice_lookup",
        "expected_invoices": ["invoice_11062.pdf"],
        "reference": (
            "The invoice for Order ID 11062 is invoice_11062.pdf.\n"
            "Order 11062 was placed by Customer ID REGGC.\n"
            "Order 11062 is dated 2018-04-30.\n"
            "Order 11062 has a total price of 508.0."
        )
    },

    # 2. Customer lookup by Order ID
    {
        "question": "Which customer placed Order ID 11062?",
        "type": "customer_by_order_id",
        "expected_invoices": ["invoice_11062.pdf"],
        "reference": (
            "Order ID 11062 was placed by Customer ID REGGC.\n"
            "The contact name for order 11062 is Maurizio Moroni.\n"
            "The customer for order 11062 is located in Reggio Emilia, Italy."
        )
    },

    # 3. Customer ID lookup with multiple relevant invoices
    {
        "question": "Find the invoice for customer DRACD",
        "type": "customer_id_multi_invoice",
        "expected_invoices": [
            "invoice_10363.pdf", "invoice_10391.pdf", "invoice_10797.pdf",
            "invoice_10825.pdf", "invoice_11036.pdf", "invoice_11067.pdf"
        ],
        "reference": (
            "invoice_10363.pdf (Order ID 10363, dated 2016-11-26) belongs to customer DRACD.\n"
            "invoice_10391.pdf (Order ID 10391, dated 2016-12-23) belongs to customer DRACD.\n"
            "invoice_10797.pdf (Order ID 10797, dated 2017-12-25) belongs to customer DRACD.\n"
            "invoice_10825.pdf (Order ID 10825, dated 2018-01-09) belongs to customer DRACD.\n"
            "invoice_11036.pdf (Order ID 11036, dated 2018-04-20) belongs to customer DRACD.\n"
            "invoice_11067.pdf (Order ID 11067, dated 2018-05-04) belongs to customer DRACD."
        )
    },

    # 4. Product lookup with multiple relevant invoices
    {
        "question": "Which invoices contain Mishi Kobe Niku?",
        "type": "product_multi_invoice",
        "expected_invoices": [
            "invoice_10420.pdf", "invoice_10515.pdf", "invoice_10687.pdf",
            "invoice_10693.pdf", "invoice_10848.pdf"
        ],
        "reference": (
            "invoice_10420.pdf (customer WELLI) contains Mishi Kobe Niku, quantity 20.\n"
            "invoice_10515.pdf (customer QUICK) contains Mishi Kobe Niku, quantity 16.\n"
            "invoice_10687.pdf (customer HUNGO) contains Mishi Kobe Niku, quantity 50.\n"
            "invoice_10693.pdf (customer WHITC) contains Mishi Kobe Niku, quantity 6.\n"
            "invoice_10848.pdf (customer CONSH) contains Mishi Kobe Niku, quantity 3."
        )
    }
]
