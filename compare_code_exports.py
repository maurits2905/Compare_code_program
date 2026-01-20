import os
import re
import argparse
import pandas as pd
from difflib import unified_diff

def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path.lower())[1]
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    raise ValueError(f"Unsupported file type: {ext}")

def normalize_code(code: str, mode: str) -> str:
    if code is None:
        return ""
    s = str(code).replace("\r\n", "\n").replace("\r", "\n")

    if mode == "none":
        return s

    if mode == "trim":
        # Trim trailing whitespace per line
        return "\n".join(line.rstrip() for line in s.split("\n"))

    if mode == "collapse_ws":
        # Collapse all whitespace blocks to single spaces (aggressive)
        return re.sub(r"\s+", " ", s).strip()

    if mode == "strip_comments_abap":
        # Simple ABAP-ish: remove full-line comments starting with *
        # and remove inline " comments (naive but useful)
        lines = []
        for line in s.split("\n"):
            raw = line.rstrip()
            if raw.lstrip().startswith("*"):
                continue
            # remove inline double-quote comments
            if '"' in raw:
                raw = raw.split('"', 1)[0].rstrip()
            lines.append(raw)
        return "\n".join(lines)

    raise ValueError(f"Unknown normalize mode: {mode}")

def build_key(df: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing key columns in file: {missing}")
    return df[key_cols].astype(str).agg("||".join, axis=1)

def filter_df(df: pd.DataFrame, author_col: str, object_col: str, persons: list[str], objects: list[str]) -> pd.DataFrame:
    out = df.copy()

    if persons:
        if author_col not in out.columns:
            raise KeyError(f"Author column '{author_col}' not found. Available: {list(out.columns)}")
        wanted = {p.lower().strip() for p in persons}
        out = out[out[author_col].fillna("").astype(str).str.lower().str.strip().isin(wanted)]

    if objects:
        if object_col not in out.columns:
            raise KeyError(f"Object column '{object_col}' not found. Available: {list(out.columns)}")
        wanted = {o.lower().strip() for o in objects}
        out = out[out[object_col].fillna("").astype(str).str.lower().str.strip().isin(wanted)]

    return out

def main():
    ap = argparse.ArgumentParser(description="Compare code snippets between two CSV/Excel exports.")
    ap.add_argument("--old", required=True, help="Old file path (.csv/.xlsx)")
    ap.add_argument("--new", required=True, help="New file path (.csv/.xlsx)")

    ap.add_argument("--object-col", default="object_name", help="Column name for object")
    ap.add_argument("--author-col", default="author", help="Column name for person/author")
    ap.add_argument("--date-col", default="date", help="Column name for date (optional, for reporting)")
    ap.add_argument("--code-col", default="code", help="Column name for code snippet")

    ap.add_argument("--key-cols", default=None,
                    help="Comma-separated key columns used to match old/new rows (default: object-col). "
                         "Example: object_name,include_name,method_name")

    ap.add_argument("--person", action="append", default=[],
                    help="Filter to these authors (repeat flag for multiple). Example: --person 'Alice' --person 'Bob'")
    ap.add_argument("--object", action="append", default=[],
                    help="Filter to these objects (repeat flag for multiple). Example: --object 'ZCL_FOO'")

    ap.add_argument("--normalize", default="trim",
                    choices=["none", "trim", "collapse_ws", "strip_comments_abap"],
                    help="How to normalize code before comparing")

    ap.add_argument("--out", default="compare_report.xlsx", help="Output report file (.xlsx or .csv)")
    ap.add_argument("--diff-dir", default="diffs", help="Directory to write per-object diff text files")
    ap.add_argument("--write-diffs", action="store_true", help="Write unified diffs per changed object")

    args = ap.parse_args()

    old_df = read_table(args.old)
    new_df = read_table(args.new)

    # Validate code column
    for df, label in [(old_df, "old"), (new_df, "new")]:
        if args.code_col not in df.columns:
            raise KeyError(f"Code column '{args.code_col}' not found in {label} file. Available: {list(df.columns)}")

    # Filter
    old_f = filter_df(old_df, args.author_col, args.object_col, args.person, args.object)
    new_f = filter_df(new_df, args.author_col, args.object_col, args.person, args.object)

    key_cols = [args.object_col] if not args.key_cols else [c.strip() for c in args.key_cols.split(",") if c.strip()]
    old_f = old_f.copy()
    new_f = new_f.copy()
    old_f["_key"] = build_key(old_f, key_cols)
    new_f["_key"] = build_key(new_f, key_cols)

    # If multiple rows per key exist, keep latest by date (if available), otherwise keep last row
    def dedupe(df: pd.DataFrame) -> pd.DataFrame:
        if args.date_col in df.columns:
            tmp = df.copy()
            tmp["_date_sort"] = pd.to_datetime(tmp[args.date_col], errors="coerce")
            tmp = tmp.sort_values(["_key", "_date_sort"]).drop_duplicates("_key", keep="last")
            return tmp.drop(columns=["_date_sort"])
        return df.drop_duplicates("_key", keep="last")

    old_u = dedupe(old_f)
    new_u = dedupe(new_f)

    merged = old_u.merge(
        new_u,
        on="_key",
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True
    )

    # Prepare comparison fields
    old_code_col = f"{args.code_col}_old"
    new_code_col = f"{args.code_col}_new"

    merged[old_code_col] = merged.get(old_code_col, "").fillna("").astype(str)
    merged[new_code_col] = merged.get(new_code_col, "").fillna("").astype(str)

    merged["code_old_norm"] = merged[old_code_col].apply(lambda s: normalize_code(s, args.normalize))
    merged["code_new_norm"] = merged[new_code_col].apply(lambda s: normalize_code(s, args.normalize))

    merged["status"] = "UNCHANGED"
    merged.loc[merged["_merge"] == "left_only", "status"] = "REMOVED_IN_NEW"
    merged.loc[merged["_merge"] == "right_only", "status"] = "ADDED_IN_NEW"
    both = merged["_merge"] == "both"
    changed = both & (merged["code_old_norm"] != merged["code_new_norm"])
    merged.loc[changed, "status"] = "CHANGED"

    # Optionally write diffs
    if args.write_diffs:
        os.makedirs(args.diff_dir, exist_ok=True)
        for _, row in merged[merged["status"] == "CHANGED"].iterrows():
            key = str(row["_key"])
            old_lines = row["code_old_norm"].splitlines(keepends=True)
            new_lines = row["code_new_norm"].splitlines(keepends=True)
            diff = unified_diff(
                old_lines, new_lines,
                fromfile="old",
                tofile="new",
                lineterm=""
            )
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[:150]
            with open(os.path.join(args.diff_dir, f"{safe}.diff.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(diff))

    # Pick useful columns for report
    report_cols = [
        "_key", "status",
        f"{args.object_col}_old", f"{args.object_col}_new",
        f"{args.author_col}_old", f"{args.author_col}_new",
        f"{args.date_col}_old", f"{args.date_col}_new",
    ]
    # Keep only those that exist
    report_cols = [c for c in report_cols if c in merged.columns]

    # Add quick metrics
    merged["old_len"] = merged["code_old_norm"].str.len()
    merged["new_len"] = merged["code_new_norm"].str.len()
    report_cols += ["old_len", "new_len"]

    report = merged[report_cols].copy()

    out_ext = os.path.splitext(args.out.lower())[1]
    if out_ext == ".csv":
        report.to_csv(args.out, index=False)
    elif out_ext in [".xlsx", ".xls"]:
        report.to_excel(args.out, index=False)
    else:
        raise ValueError("Output must be .csv or .xlsx")

    print("Done.")
    print("Rows in old (filtered):", len(old_f), "=> unique keys:", len(old_u))
    print("Rows in new (filtered):", len(new_f), "=> unique keys:", len(new_u))
    print(report["status"].value_counts().to_string())
    print("Report:", args.out)
    if args.write_diffs:
        print("Diffs:", args.diff_dir)

if __name__ == "__main__":
    main()
