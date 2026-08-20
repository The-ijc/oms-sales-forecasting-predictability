import json
import re
from pathlib import Path

TARGET = Path("data/product_knowledge/products.json")
PATCH = Path("product_knowledge_enrichment_first_batch.json")
BACKUP = TARGET.with_suffix(".json.bak")

def norm(model):
    return re.sub(r"[^A-Z0-9]", "", str(model or "").upper())

products = json.loads(TARGET.read_text(encoding="utf-8"))
patches = json.loads(PATCH.read_text(encoding="utf-8"))

BACKUP.write_text(
    json.dumps(products, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

by_norm = {norm(p.get("model") or p.get("normalized_model")): p for p in products}

updated = 0
created = 0

for patch in patches:
    key = norm(patch.get("model"))
    if not key:
        continue

    current = by_norm.get(key)
    if current is None:
        current = {
            "model": patch.get("model", ""),
            "normalized_model": key,
            "brand": "",
            "category": "",
            "subcategory": "",
            "product_name": "",
            "launch_date": "",
            "specifications": {},
            "features": [],
            "description": "",
            "sources": [],
        }
        products.append(current)
        by_norm[key] = current
        created += 1

    changed = False

    # Only fill empty scalar fields; keep SQLite-derived category/subcategory/launch_date.
    for field in ("brand", "product_name", "description"):
        if not str(current.get(field) or "").strip() and str(patch.get(field) or "").strip():
            current[field] = patch[field]
            changed = True

    if not current.get("specifications") and patch.get("specifications"):
        current["specifications"] = patch["specifications"]
        changed = True

    if not current.get("features") and patch.get("features"):
        current["features"] = patch["features"]
        changed = True

    if not current.get("sources") and patch.get("sources"):
        current["sources"] = patch["sources"][:1]
        changed = True

    if changed:
        updated += 1

TARGET.write_text(
    json.dumps(products, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print(f"updated={updated}, created={created}, total={len(products)}")
print(f"backup={BACKUP}")
