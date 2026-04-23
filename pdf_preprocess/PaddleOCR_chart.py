from paddlex import create_model
import os
import csv
import json


def extract_table_text(prediction_result):
    """Try multiple strategies to extract the table string from a prediction result.

    Expected structures (best-effort):
    - Object with attribute .res -> dict containing key 'result' (string)
    - Dict with key 'res' -> dict containing key 'result' (string)
    - Object with .to_json() returning a JSON string containing the above
    - Fallback: inspect __dict__ for 'res' -> 'result'
    """
    # 1) Common case: object with .res dict
    data = getattr(prediction_result, "res", None)
    if isinstance(data, dict):
        result_text = data.get("result")
        if isinstance(result_text, str) and result_text.strip():
            return result_text

    # 2) If it's already a dict
    if isinstance(prediction_result, dict):
        inner = prediction_result.get("res", prediction_result)
        if isinstance(inner, dict):
            result_text = inner.get("result")
            if isinstance(result_text, str) and result_text.strip():
                return result_text

    # 3) Try to_json() if available
    to_json_method = getattr(prediction_result, "to_json", None)
    if callable(to_json_method):
        try:
            obj = json.loads(to_json_method())
            inner = obj.get("res", obj)
            if isinstance(inner, dict):
                result_text = inner.get("result")
                if isinstance(result_text, str) and result_text.strip():
                    return result_text
        except Exception:
            pass

    # 4) Fallback: inspect __dict__
    try:
        d = prediction_result.__dict__
        if isinstance(d, dict):
            inner = d.get("res", d)
            if isinstance(inner, dict):
                result_text = inner.get("result")
                if isinstance(result_text, str) and result_text.strip():
                    return result_text
    except Exception:
        pass

    return None


def parse_table_to_rows(table_text):
    """Parse table_text into a list of list of cells by splitting lines and '|'."""
    lines = [ln.strip() for ln in table_text.strip().splitlines() if ln.strip()]
    raw_rows = []
    for line in lines:
        # Split on '|' and strip each cell
        cells = [cell.strip() for cell in line.split("|")]
        raw_rows.append(cells)

    # Normalize row lengths to the max column count
    col_count = max((len(r) for r in raw_rows), default=0)
    rows = []
    for r in raw_rows:
        if len(r) < col_count:
            r = r + [""] * (col_count - len(r))
        elif len(r) > col_count:
            r = r[:col_count]
        rows.append(r)
    return rows


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def write_markdown(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not rows:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("")
        return

    header = rows[0]
    col_count = len(header)
    separator = ["---"] * col_count
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("| " + " | ".join(separator) + " |\n")
        for r in rows[1:]:
            # Pad/truncate to header length for a proper markdown table
            if len(r) < col_count:
                r = r + [""] * (col_count - len(r))
            elif len(r) > col_count:
                r = r[:col_count]
            f.write("| " + " | ".join(r) + " |\n")


model = create_model('PP-Chart2Table')
# Load the image from ../Data/Charts folder
for file in os.listdir("../Data/Charts"):
    if file.endswith(".png"):
        results = model.predict(
            input={"image": "../Data/Charts/" + file},
            batch_size=1
        )
        for res in results:
            # Print for visibility/debugging
            res.print()

            table_text = extract_table_text(res)
            if not table_text:
                print(f"[WARN] Could not extract table text for {file}.")
                continue

            rows = parse_table_to_rows(table_text)

            # Write CSV and Markdown instead of JSON
            base_out = f"./output_charts/{file}"
            csv_path = base_out + ".csv"
            md_path = base_out + ".md"
            write_csv(rows, csv_path)
            write_markdown(rows, md_path)
            print(f"[INFO] Saved CSV: {csv_path}")
            print(f"[INFO] Saved Markdown: {md_path}")