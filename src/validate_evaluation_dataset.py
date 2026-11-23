"""
Validate the generated evaluation datasets against the raw invoice text.

    python -m src.validate_evaluation_dataset

Independent of the generator: it does NOT reuse its parser. Ground truth is
recomputed from data/processed/extracted_invoices.json with direct text
matching on each record's `criteria`, and every reference statement is
checked against the invoice it names. Exit code 1 if any check fails.
"""

import os
import re
import sys
import json
from collections import Counter, defaultdict

from .evaluation_dataset import evaluation_data, DATASET_VERSION
from .ragas_evaluation_dataset import ragas_evaluation_data
from .generate_evaluation_dataset import CATEGORY_TARGETS


EXTRACTED_PATH = "data/processed/extracted_invoices.json"
PDF_DIR = "data/invoices"
TOL = 0.005


def load_raw_text():
    with open(EXTRACTED_PATH, "r", encoding="utf-8") as file:
        pages = json.load(file)
    text = defaultdict(str)
    for page in sorted(pages, key=lambda p: (p["source"], p["page_number"])):
        text[page["source"]] += page["text"] + "\n"
    return dict(text)


# --- raw-text helpers (deliberately simple and separate from the generator)

def items(text):
    # (name, quantity, unit_price) for every product row
    rows = re.findall(r"(?m)^\d{1,3}$\n^(.+)$\n^(\d+)$\n^(\d+(?:\.\d+)?)$", text)
    return [(name, int(q), float(p)) for name, q, p in rows
            if re.search(r"[^\W\d_]", name)]


def total(text):
    return float(re.search(r"TotalPrice\n([\d.]+)", text).group(1))


def has_line(text, label, value):
    return f"{label}\n{value}\n" in text or f"{label} {value}\n" in text


def recompute(criteria, raw):
    """Set of invoices matching the criteria, from raw text only."""
    kind = criteria["type"]
    value = criteria.get("value")
    totals = {s: total(t) for s, t in raw.items()}

    tests = {
        "order_id": lambda t: f"Order ID: {value}\n" in t,
        "customer_id": lambda t: f"Customer ID: {value}\n" in t,
        "contact_name": lambda t: f"Contact Name:\n{value}\n" in t,
        "product": lambda t: any(n == value for n, _, _ in items(t)),
        "product_pair": lambda t: set(value) <= {n for n, _, _ in items(t)},
        "unit_price_above": lambda t: any(p > value for *_, p in items(t)),
        "total_above": lambda t: total(t) > value,
        "max_total": lambda t: abs(total(t) - max(totals.values())) < TOL,
        "min_total": lambda t: abs(total(t) - min(totals.values())) < TOL,
        "country": lambda t: f"Country:\n{value}\n" in t,
        "line_items_at_least": lambda t: len(items(t)) >= value,
        "date_prefix": lambda t: f"Order Date: {value}" in t,
        "quantity_at_least": lambda t: any(q >= value for _, q, _ in items(t)),
        "total_price": lambda t: f"Order ID: {criteria['order_id']}\n" in t,
        "quantity": lambda t: f"Order ID: {criteria['order_id']}\n" in t,
        "unit_price": lambda t: f"Order ID: {criteria['order_id']}\n" in t,
    }
    if kind not in tests:
        raise ValueError(f"unknown criteria type {kind}")
    return {s for s, t in raw.items() if tests[kind](t)}


def numbers_in(text):
    return [float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", text)]


def statement_supported(statement, invoice_text):
    """Every ID, date and number in the statement occurs in the invoice."""
    problems = []
    for order_id in re.findall(r"Order ID (\d+)", statement):
        if f"Order ID: {order_id}\n" not in invoice_text:
            problems.append(f"order {order_id}")
    for cid in re.findall(r"customer ([A-Z]{4,6})\b", statement):
        if f"Customer ID: {cid}\n" not in invoice_text:
            problems.append(f"customer {cid}")
    for date in re.findall(r"\d{4}-\d{2}-\d{2}", statement):
        if date not in invoice_text:
            problems.append(f"date {date}")
    # Derived counts are checked by counting rows, not as printed numbers
    for n in re.findall(r"has (\d+) product line items", statement):
        if int(n) != len(items(invoice_text)):
            problems.append(f"line-item count {n}")
    source_numbers = numbers_in(invoice_text)
    cleaned = re.sub(r"invoice_\d+\.pdf|\d{4}-\d{2}-\d{2}|Order ID \d+"
                     r"|has \d+ product line items",
                     " ", statement)
    for number in numbers_in(cleaned):
        if not any(abs(number - n) < TOL for n in source_numbers):
            problems.append(f"number {number}")
    return problems


def main():

    raw = load_raw_text()
    pdfs = {f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")}
    failures, warnings = [], []

    def check(ok, message):
        if not ok:
            failures.append(message)

    print(f"Validating dataset {DATASET_VERSION}: "
          f"{len(evaluation_data)} retrieval records, "
          f"{len(ragas_evaluation_data)} RAGAS records\n")

    # Source coverage
    check(pdfs == set(raw), f"PDFs ({len(pdfs)}) and extracted invoices "
          f"({len(raw)}) differ: {sorted(pdfs ^ set(raw))[:5]}")

    # IDs, duplicates, pairing of the two datasets
    ids = [r["id"] for r in evaluation_data]
    check(len(ids) == len(set(ids)), "duplicate question IDs")
    questions = [r["question"].casefold() for r in evaluation_data]
    dupes = [q for q, c in Counter(questions).items() if c > 1]
    check(not dupes, f"duplicate questions: {dupes}")
    ragas_by_id = {r["id"]: r for r in ragas_evaluation_data}
    check(set(ragas_by_id) == set(ids), "retrieval and RAGAS IDs differ")

    # Category counts
    counts = Counter(r["category"] for r in evaluation_data)
    for category, target in CATEGORY_TARGETS.items():
        if counts[category] < target:
            warnings.append(f"{category}: {counts[category]}/{target}")

    numeric_checked = 0
    statements_checked = 0

    for rec in evaluation_data:

        rid = rec["id"]
        expected = set(rec["expected_invoices"])
        ragas = ragas_by_id.get(rid, {})
        reference = ragas.get("reference", "")

        # Files exist
        missing = [s for s in expected if s not in pdfs or s not in raw]
        check(not missing, f"{rid}: expected invoices not found {missing}")

        # Same question and ground truth in both datasets
        check(ragas.get("user_input") == rec["question"],
              f"{rid}: question differs between datasets")
        check(set(ragas.get("expected_invoices", [])) == expected,
              f"{rid}: expected_invoices differ between datasets")

        # Completeness: recompute the full relevant set from raw text
        truth = recompute(rec["criteria"], raw)
        check(truth == expected,
              f"{rid}: expected_invoices wrong - missing "
              f"{sorted(truth - expected)[:5]}, extra "
              f"{sorted(expected - truth)[:5]}")

        # Reference non-empty and names exactly the expected invoices
        check(bool(reference.strip()), f"{rid}: empty reference")
        named = set(re.findall(r"invoice_\d+\.pdf", reference))
        check(named == expected,
              f"{rid}: reference names {len(named)} invoices, expected "
              f"{len(expected)}")

        # Count statements ("N invoices match", "has N invoices")
        for n in re.findall(r"(\d+) invoices (?:match|\.)", reference) + \
                re.findall(r"has (\d+) invoices", reference) + \
                re.findall(r"appears in (\d+) invoices", reference):
            check(int(n) == len(expected),
                  f"{rid}: reference says {n} invoices, ground truth "
                  f"{len(expected)}")

        # Every statement that names an invoice is supported by THAT invoice
        for statement in reference.split("\n"):
            for source in re.findall(r"invoice_\d+\.pdf", statement):
                statements_checked += 1
                problems = statement_supported(statement, raw.get(source, ""))
                check(not problems,
                      f"{rid}: '{statement[:70]}' not supported by "
                      f"{source}: {problems}")

        # Statements without a filename but with an order ID
        for statement in reference.split("\n"):
            if "invoice_" in statement:
                continue
            for order_id in re.findall(r"(?:Order ID|Order|order) (\d{5})",
                                       statement):
                source = f"invoice_{order_id}.pdf"
                statements_checked += 1
                problems = statement_supported(statement, raw.get(source, ""))
                check(not problems,
                      f"{rid}: '{statement[:70]}' not supported: {problems}")

        # Numerical answers equal the invoice values
        crit = rec["criteria"]
        if crit["type"] in ("total_price", "quantity", "unit_price"):
            numeric_checked += 1
            text = raw[f"invoice_{crit['order_id']}.pdf"]
            if crit["type"] == "total_price":
                actual = total(text)
            else:
                row = [r for r in items(text) if r[0] == crit["product"]]
                check(len(row) == 1, f"{rid}: product row not unique")
                actual = row[0][1] if crit["type"] == "quantity" else row[0][2]
            check(abs(actual - crit["value"]) < TOL,
                  f"{rid}: value {crit['value']} != invoice {actual}")
            check(any(abs(n - actual) < TOL for n in numbers_in(reference)),
                  f"{rid}: reference does not state {actual}")

    # Report
    print("Category counts:")
    for category, target in CATEGORY_TARGETS.items():
        print(f"  {category:<22} {counts[category]:>3} / {target}")
    sizes = [len(r["expected_invoices"]) for r in evaluation_data]
    print(f"\nRelevant invoices per question: min {min(sizes)}, "
          f"max {max(sizes)}, single-answer {sizes.count(1)}, "
          f"multi-answer {len(sizes) - sizes.count(1)}")
    print(f"Reference statements checked against source invoices: "
          f"{statements_checked}")
    print(f"Numerical answers checked: {numeric_checked}")

    for w in warnings:
        print("WARNING:", w)

    if failures:
        print(f"\nFAILED: {len(failures)} problem(s)")
        for f in failures:
            print("  -", f)
        sys.exit(1)

    print("\nPASSED: all checks")


if __name__ == "__main__":
    main()
