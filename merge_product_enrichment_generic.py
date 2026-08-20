import argparse
import json
import re
from pathlib import Path

def norm(model):
    return re.sub(r"[^A-Z0-9]", "", str(model or "").upper())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patch_file", help="Enrichment JSON file")
    parser.add_argument(
        "--target",
        default="data/product_knowledge/products.json",
        help="Target products.json"
    )
    args = parser.parse_args()

    target = Path(args.target)
    patch_file = Path(args.patch_file)

    products = json.loads(target.read_text(encoding="utf-8"))
    patches = json.loads(patch_file.read_text(encoding="utf-8"))

    backup = target.with_name(target.stem + ".before_batch_merge.json")
    backup.write_text(
        json.dumps(products, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    by_model = {
        norm(p.get("normalized_model") or p.get("model")): p
        for p in products
    }

    updated = 0
    missing = []

    for patch in patches:
        key = norm(patch.get("model"))
        current = by_model.get(key)

        if current is None:
            missing.append(patch.get("model"))
            continue

        changed = False

        # Fill only empty knowledge fields.
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

    target.write_text(
        json.dumps(products, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"UPDATED={updated}")
    print(f"MISSING={len(missing)}")
    if missing:
        print("MISSING_MODELS=" + ",".join(map(str, missing)))
    print(f"TOTAL={len(products)}")
    print(f"BACKUP={backup}")

if __name__ == "__main__":
    main()
