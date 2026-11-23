"""
Generate the invoice evaluation datasets from the extracted invoice data.

    python -m src.generate_evaluation_dataset

Reads   data/processed/extracted_invoices.json   (PyMuPDF output, not re-extracted)
Writes  src/evaluation_dataset.py                 document-level retrieval ground truth
        src/ragas_evaluation_dataset.py           question + reference for RAGAS
        data/evaluation/evaluation_dataset.json
        data/evaluation/ragas_evaluation_dataset.json

Every expected invoice set and every reference value is computed from the
invoice records; nothing is typed by hand. Each record also carries a
machine-readable `criteria` so src/validate_evaluation_dataset.py can
recompute the ground truth independently from the raw text.
"""

import os
import re
import json
import random
import hashlib
import pprint
from collections import defaultdict


DATASET_VERSION = "invoice-eval-v1"
SEED = 42

EXTRACTED_PATH = "data/processed/extracted_invoices.json"
OUT_DIR_JSON = "data/evaluation"
OUT_EVAL_PY = "src/evaluation_dataset.py"
OUT_RAGAS_PY = "src/ragas_evaluation_dataset.py"

CATEGORY_TARGETS = {
    "order_id_lookup": 10,
    "customer_id_lookup": 10,
    "product_lookup": 10,
    "customer_name_lookup": 5,
    "numerical": 5,
    "natural_language": 10,
}

ID_PREFIX = {
    "order_id_lookup": "ORD",
    "customer_id_lookup": "CUS",
    "product_lookup": "PRD",
    "customer_name_lookup": "NAM",
    "numerical": "NUM",
    "natural_language": "NLQ",
}


# ---------------------------------------------------------------------------
# 1. Parse invoices into structured records
# ---------------------------------------------------------------------------

LINE_ITEM_RE = re.compile(
    r"\n(\d{1,3})\n([^\n]*[^\W\d_][^\n]*)\n(\d+)\n(\d+(?:\.\d+)?)(?=\n)"
)


def field(text, label, next_label):
    match = re.search(
        re.escape(label) + r"\n(.*?)\n" + re.escape(next_label), text, re.S)
    return match.group(1).strip() if match else None


def parse_invoices(path=EXTRACTED_PATH):

    with open(path, "r", encoding="utf-8") as file:
        pages = json.load(file)

    text_by_source = defaultdict(str)
    for page in sorted(pages, key=lambda p: (p["source"], p["page_number"])):
        text_by_source[page["source"]] += page["text"] + "\n"

    invoices, problems = {}, []

    for source, text in sorted(text_by_source.items()):

        order_id = re.search(r"Order ID:\s*(\d+)", text)
        customer_id = re.search(r"Customer ID:\s*(\w+)", text)
        date = re.search(r"Order Date:\s*(\d{4}-\d{2}-\d{2})", text)
        total = re.search(r"TotalPrice\n(\d+(?:\.\d+)?)", text)

        record = {
            "source": source,
            "order_id": order_id.group(1) if order_id else None,
            "customer_id": customer_id.group(1) if customer_id else None,
            "order_date": date.group(1) if date else None,
            "contact_name": field(text, "Contact Name:", "Address:"),
            "city": field(text, "City:", "Postal Code:"),
            "country": field(text, "Country:", "Phone:"),
            "total_price": round(float(total.group(1)), 2) if total else None,
            "items": [
                {"product_id": int(pid), "name": name.strip(),
                 "quantity": int(qty), "unit_price": float(price)}
                for pid, name, qty, price in LINE_ITEM_RE.findall(text)
            ],
        }

        # Exclude any invoice we cannot parse confidently
        missing = [k for k, v in record.items() if v in (None, "", [])]
        line_sum = round(sum(i["quantity"] * i["unit_price"]
                             for i in record["items"]), 2)
        if missing:
            problems.append(f"{source}: missing {missing}")
        elif abs(line_sum - record["total_price"]) > 0.01:
            problems.append(f"{source}: line items sum {line_sum} != "
                            f"TotalPrice {record['total_price']}")
        else:
            invoices[source] = record

    return invoices, problems


def money(value):
    return f"{value:.2f}"


def invoice_line(inv, extra=""):
    # One verifiable statement per invoice (RAGAS ContextRecall scores the
    # share of statements supported, so lists are never one long sentence)
    return (f"{inv['source']} (Order ID {inv['order_id']}, customer "
            f"{inv['customer_id']}, dated {inv['order_date']}){extra}.")


# ---------------------------------------------------------------------------
# 2. Question builders (one per category)
# ---------------------------------------------------------------------------

def record(category, question, invoices, reference, criteria):
    expected = sorted(inv["source"] for inv in invoices)
    return {
        "category": category,
        "question": question,
        "expected_invoices": expected,
        "source_order_ids": sorted(inv["order_id"] for inv in invoices),
        "reference": reference,
        "criteria": criteria,
    }


def order_id_questions(invoices, rng, n, used):

    pool = sorted(s for s in invoices if s not in used)
    chosen = rng.sample(pool, n)
    used.update(chosen)
    out = []

    for i, source in enumerate(chosen):
        inv = invoices[source]
        criteria = {"type": "order_id", "value": inv["order_id"]}

        if i % 2 == 0:
            question = f"Find the invoice with Order ID {inv['order_id']}."
            reference = (
                f"The invoice for Order ID {inv['order_id']} is {source}.\n"
                f"Order {inv['order_id']} was placed by customer "
                f"{inv['customer_id']} ({inv['contact_name']}).\n"
                f"Order {inv['order_id']} is dated {inv['order_date']}.\n"
                f"Order {inv['order_id']} has a total price of "
                f"{money(inv['total_price'])}."
            )
        else:
            question = f"Which customer placed Order ID {inv['order_id']}?"
            reference = (
                f"Order ID {inv['order_id']} ({source}) was placed by "
                f"customer {inv['customer_id']}.\n"
                f"The contact name for order {inv['order_id']} is "
                f"{inv['contact_name']}.\n"
                f"The customer for order {inv['order_id']} is located in "
                f"{inv['city']}, {inv['country']}."
            )

        out.append(record("order_id_lookup", question, [inv], reference,
                          criteria))
    return out


def by_customer(invoices):
    groups = defaultdict(list)
    for inv in invoices.values():
        groups[inv["customer_id"]].append(inv)
    return groups


def customer_reference(customer_id, group, lead):
    inv0 = group[0]
    lines = [lead.format(cid=customer_id, name=inv0["contact_name"],
                         n=len(group), city=inv0["city"],
                         country=inv0["country"])]
    lines += [invoice_line(inv) for inv in
              sorted(group, key=lambda x: x["source"])]
    return "\n".join(lines)


def customer_id_questions(invoices, rng, n, used_customers):

    groups = by_customer(invoices)
    pool = sorted(c for c in groups if c not in used_customers)
    chosen = rng.sample(pool, n)
    used_customers.update(chosen)
    out = []

    templates = ["Find all invoices belonging to customer {cid}.",
                 "Which invoices belong to customer {cid}?"]

    for i, cid in enumerate(chosen):
        group = groups[cid]
        out.append(record(
            "customer_id_lookup",
            templates[i % 2].format(cid=cid),
            group,
            customer_reference(
                cid, group,
                "Customer {cid} (contact {name}, {city}, {country}) has "
                "{n} invoices."),
            {"type": "customer_id", "value": cid},
        ))
    return out


def customer_name_questions(invoices, rng, n, used_customers):

    groups = by_customer(invoices)
    pool = sorted(c for c in groups if c not in used_customers)
    chosen = rng.sample(pool, n)
    used_customers.update(chosen)
    out = []

    for cid in chosen:
        group = groups[cid]
        name = group[0]["contact_name"]
        out.append(record(
            "customer_name_lookup",
            f"Find invoices associated with {name}.",
            group,
            customer_reference(
                cid, group,
                "{name} is the contact for customer {cid}, which has "
                "{n} invoices."),
            {"type": "contact_name", "value": name},
        ))
    return out


def by_product(invoices):
    groups = defaultdict(list)
    for inv in invoices.values():
        for item in inv["items"]:
            groups[item["name"]].append((inv, item))
    return groups


def product_questions(invoices, rng, n):

    groups = by_product(invoices)
    # Stratified: half from rarer products (<= 15 invoices), half from the
    # rest, so the set has both small and large answer sets
    names = sorted(groups)
    rare = [p for p in names if len(groups[p]) <= 15]
    common = [p for p in names if len(groups[p]) > 15]
    chosen = rng.sample(rare, n // 2) + rng.sample(common, n - n // 2)
    out = []

    templates = ["Which invoices contain {p}?",
                 "Find invoices containing {p}."]

    for i, product in enumerate(chosen):
        pairs = sorted(groups[product], key=lambda x: x[0]["source"])
        lines = [f"{product} appears in {len(pairs)} invoices."]
        lines += [invoice_line(inv, f" contains {product}, quantity "
                               f"{item['quantity']}") for inv, item in pairs]
        out.append(record(
            "product_lookup",
            templates[i % 2].format(p=product),
            [inv for inv, _ in pairs],
            "\n".join(lines),
            {"type": "product", "value": product},
        ))
    return out


def numerical_questions(invoices, rng, n, used):

    pool = sorted(s for s in invoices if s not in used)
    chosen = rng.sample(pool, n)
    used.update(chosen)
    out = []

    for i, source in enumerate(chosen):
        inv = invoices[source]
        kind = ("total_price", "quantity", "unit_price")[i % 3]

        if kind == "total_price":
            question = f"What is the total price of Order ID {inv['order_id']}?"
            value = inv["total_price"]
            reference = (f"The total price of Order ID {inv['order_id']} "
                         f"({source}) is {money(value)}.")
            criteria = {"type": "total_price", "order_id": inv["order_id"],
                        "value": value}
        else:
            item = rng.choice(inv["items"])
            if kind == "quantity":
                question = (f"What quantity of {item['name']} was ordered in "
                            f"Order ID {inv['order_id']}?")
                value = item["quantity"]
                reference = (f"Order ID {inv['order_id']} ({source}) "
                             f"includes {value} units of {item['name']}.")
            else:
                question = (f"What is the unit price of {item['name']} in "
                            f"Order ID {inv['order_id']}?")
                value = item["unit_price"]
                reference = (f"The unit price of {item['name']} in Order ID "
                             f"{inv['order_id']} ({source}) is "
                             f"{money(value)}.")
            criteria = {"type": kind, "order_id": inv["order_id"],
                        "product": item["name"], "value": value}

        out.append(record("numerical", question, [inv], reference, criteria))
    return out


def natural_language_questions(invoices, rng, excluded):

    all_inv = sorted(invoices.values(), key=lambda x: x["source"])
    max_total = max(i["total_price"] for i in all_inv)
    min_total = min(i["total_price"] for i in all_inv)

    # Product pair bought together in 2-8 invoices (seeded choice)
    product_sets = {i["source"]: {it["name"] for it in i["items"]}
                    for i in all_inv}
    products = sorted({p for s in product_sets.values() for p in s})
    pair = None
    for _ in range(500):
        a, b = sorted(rng.sample(products, 2))
        hits = [s for s, ps in product_sets.items() if a in ps and b in ps]
        if 2 <= len(hits) <= 8:
            pair = (a, b)
            break

    specs = [
        ("Which invoices contain expensive products, with a unit price "
         "above 200?",
         {"type": "unit_price_above", "value": 200},
         lambda i: any(it["unit_price"] > 200 for it in i["items"]),
         lambda i: " includes " + ", ".join(
             f"{it['name']} at {money(it['unit_price'])}"
             for it in i["items"] if it["unit_price"] > 200)),
        ("Which invoices have a total price above 10000?",
         {"type": "total_above", "value": 10000},
         lambda i: i["total_price"] > 10000,
         lambda i: f" has a total price of {money(i['total_price'])}"),
        ("Which invoice has the highest total price?",
         {"type": "max_total"},
         lambda i: i["total_price"] == max_total,
         lambda i: f" has the highest total price, "
                   f"{money(i['total_price'])}"),
        ("Which invoice has the lowest total price?",
         {"type": "min_total"},
         lambda i: i["total_price"] == min_total,
         lambda i: f" has the lowest total price, "
                   f"{money(i['total_price'])}"),
        ("Which invoices are from customers in Norway?",
         {"type": "country", "value": "Norway"},
         lambda i: i["country"] == "Norway",
         lambda i: f" is from {i['city']}, Norway"),
        ("Which invoices are from customers in Poland?",
         {"type": "country", "value": "Poland"},
         lambda i: i["country"] == "Poland",
         lambda i: f" is from {i['city']}, Poland"),
        ("Which invoices contain six or more product line items?",
         {"type": "line_items_at_least", "value": 6},
         lambda i: len(i["items"]) >= 6,
         lambda i: f" has {len(i['items'])} product line items"),
        ("Which invoices were placed in May 2018?",
         {"type": "date_prefix", "value": "2018-05"},
         lambda i: i["order_date"].startswith("2018-05"),
         lambda i: ""),
        ("Which invoices include an order line with a quantity of 100 or "
         "more?",
         {"type": "quantity_at_least", "value": 100},
         lambda i: any(it["quantity"] >= 100 for it in i["items"]),
         lambda i: " includes " + ", ".join(
             f"{it['quantity']} units of {it['name']}"
             for it in i["items"] if it["quantity"] >= 100)),
    ]

    if pair:
        a, b = pair
        specs.append((
            f"Which invoices contain both {a} and {b}?",
            {"type": "product_pair", "value": [a, b]},
            lambda i: {a, b} <= {it["name"] for it in i["items"]},
            lambda i: f" contains both {a} and {b}"))
    else:
        excluded.append("natural_language: no product pair found in 2-8 "
                        "invoices within 500 seeded draws")

    out = []
    for question, criteria, predicate, detail in specs:
        matches = [i for i in all_inv if predicate(i)]
        if not matches:
            excluded.append(f"natural_language: '{question}' has no "
                            f"matching invoice")
            continue
        lines = [f"{len(matches)} invoices match."] if len(matches) > 1 \
            else []
        lines += [invoice_line(i, detail(i)) for i in matches]
        out.append(record("natural_language", question, matches,
                          "\n".join(lines), criteria))
    return out


# ---------------------------------------------------------------------------
# 3. Assemble, de-duplicate, assign stable IDs, export
# ---------------------------------------------------------------------------

def generate():

    invoices, problems = parse_invoices()
    rng = random.Random(SEED)
    excluded = list(problems)

    used_invoices, used_customers = set(), set()
    records = []
    records += order_id_questions(
        invoices, rng, CATEGORY_TARGETS["order_id_lookup"], used_invoices)
    records += customer_id_questions(
        invoices, rng, CATEGORY_TARGETS["customer_id_lookup"], used_customers)
    records += product_questions(
        invoices, rng, CATEGORY_TARGETS["product_lookup"])
    records += customer_name_questions(
        invoices, rng, CATEGORY_TARGETS["customer_name_lookup"],
        used_customers)
    records += numerical_questions(
        invoices, rng, CATEGORY_TARGETS["numerical"], used_invoices)
    records += natural_language_questions(invoices, rng, excluded)

    # De-duplicate questions (case-insensitive)
    seen, unique = set(), []
    for rec in records:
        key = rec["question"].casefold()
        if key in seen:
            excluded.append(f"duplicate question dropped: {rec['question']}")
            continue
        seen.add(key)
        unique.append(rec)

    counters = defaultdict(int)
    for rec in unique:
        counters[rec["category"]] += 1
        rec["id"] = f"{ID_PREFIX[rec['category']]}-{counters[rec['category']]:02d}"

    for category, target in CATEGORY_TARGETS.items():
        if counters[category] < target:
            excluded.append(f"{category}: {counters[category]} of {target} "
                            f"questions (not enough verifiable source data)")

    with open(EXTRACTED_PATH, "rb") as file:
        fingerprint = hashlib.sha256(file.read()).hexdigest()[:16]

    meta = {
        "dataset_version": DATASET_VERSION,
        "seed": SEED,
        "source_file": EXTRACTED_PATH,
        "source_sha256_16": fingerprint,
        "invoices_parsed": len(invoices),
        "questions": len(unique),
        "per_category": dict(counters),
        "excluded": excluded,
    }

    return unique, meta


def write_python(path, variable, rows, meta, doc):

    with open(path, "w", encoding="utf-8") as file:
        file.write(f'"""\n{doc}\n\nGENERATED by src/generate_evaluation_dataset.py '
                   f'- do not edit by hand.\nRegenerate with: '
                   f'python -m src.generate_evaluation_dataset\n"""\n\n')
        file.write(f"DATASET_VERSION = {meta['dataset_version']!r}\n")
        file.write(f"SEED = {meta['seed']!r}\n")
        file.write(f"SOURCE_SHA256_16 = {meta['source_sha256_16']!r}\n\n")
        file.write(f"{variable} = ")
        file.write(pprint.pformat(rows, width=100, sort_dicts=False))
        file.write("\n")


def export(records, meta):

    os.makedirs(OUT_DIR_JSON, exist_ok=True)

    # Document-level retrieval ground truth
    retrieval_rows = [{
        "id": r["id"],
        "category": r["category"],
        "question": r["question"],
        "expected_invoices": r["expected_invoices"],
        "source_order_ids": r["source_order_ids"],
        "criteria": r["criteria"],
    } for r in records]

    # RAGAS context evaluation: user_input + reference. Contexts are NOT
    # stored; they are retrieved from FAISS at evaluation time.
    ragas_rows = [{
        "id": r["id"],
        "category": r["category"],
        "user_input": r["question"],
        "question": r["question"],
        "reference": r["reference"],
        "expected_invoices": r["expected_invoices"],
    } for r in records]

    write_python(
        OUT_EVAL_PY, "evaluation_data", retrieval_rows, meta,
        "Document-level retrieval ground truth (Hit@K, Precision@K, "
        "Recall@K, MRR@K).\nexpected_invoices lists EVERY relevant invoice.")
    write_python(
        OUT_RAGAS_PY, "ragas_evaluation_data", ragas_rows, meta,
        "RAGAS context evaluation (Context Precision, Context Recall).\n"
        "Retrieved contexts are collected from FAISS during evaluation.")

    for name, rows in (("evaluation_dataset", retrieval_rows),
                       ("ragas_evaluation_dataset", ragas_rows)):
        with open(f"{OUT_DIR_JSON}/{name}.json", "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "data": rows}, f, indent=2,
                      ensure_ascii=False)


if __name__ == "__main__":

    records, meta = generate()
    export(records, meta)

    print(f"Dataset {meta['dataset_version']} (seed {meta['seed']}, "
          f"source {meta['source_sha256_16']})")
    print(f"Invoices parsed: {meta['invoices_parsed']}")
    print(f"Questions: {meta['questions']}")
    for category, count in meta["per_category"].items():
        print(f"  {category:<22} {count:>3} / "
              f"{CATEGORY_TARGETS[category]}")
    if meta["excluded"]:
        print("Excluded / notes:")
        for note in meta["excluded"]:
            print("  -", note)
    print(f"\nWrote {OUT_EVAL_PY}, {OUT_RAGAS_PY} and {OUT_DIR_JSON}/*.json")
