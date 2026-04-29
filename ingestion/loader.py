
import pdfplumber
import docx
import pandas as pd


# ---------------- CLEANING ----------------
def clean_text(text: str):
    if text is None:
        return None

    text = str(text).strip()
    return text if text else None


# ---------------- PDF ----------------
def read_pdf(file):
    try:
        file.seek(0)

        pages = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t and t.strip():
                    pages.append(t.strip())

        return clean_text("\n".join(pages))

    except Exception as e:
        print("PDF error:", e)
        return None


# ---------------- DOCX ----------------
def read_docx(file):
    try:
        file.seek(0)

        doc = docx.Document(file)
        paragraphs = [
            p.text.strip()
            for p in doc.paragraphs
            if p.text and p.text.strip()
        ]

        return clean_text("\n".join(paragraphs))

    except Exception as e:
        print("DOCX error:", e)
        return None


# ---------------- CSV ----------------
def read_csv(file):
    try:
        file.seek(0)

        # Try default first
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin-1", sep=None, engine="python")

        if df is None or df.empty:
            return None

        rows = []
        columns = df.columns.tolist()

        for _, row in df.iterrows():
            values = [
                f"{col}: {str(val).strip()}"
                for col, val in zip(columns, row.values)
                if pd.notna(val)
            ]

            row_text = " | ".join(values)

            if row_text:
                rows.append(row_text)

        return clean_text("\n".join(rows))

    except Exception as e:
        print("CSV error:", e)
        return None


# ---------------- XLSX ----------------
def read_xlsx(file):
    try:
        file.seek(0)

        # Read ALL sheets
        sheets = pd.read_excel(file, engine="openpyxl", sheet_name=None)

        all_rows = []

        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue

            columns = df.columns.tolist()

            for _, row in df.iterrows():
                values = [
                    f"{col}: {str(val).strip()}"
                    for col, val in zip(columns, row.values)
                    if pd.notna(val)
                ]

                row_text = f"[Sheet: {sheet_name}] " + " | ".join(values)

                if row_text.strip():
                    all_rows.append(row_text)

        return clean_text("\n".join(all_rows))

    except Exception as e:
        print("XLSX error:", e)
        return None


# ---------------- MAIN LOADER ----------------
def load_file(file):
    if file is None:
        return None

    try:
        file_type = file.type

        if file_type == "application/pdf":
            return read_pdf(file)

        elif file_type.endswith("wordprocessingml.document"):
            return read_docx(file)

        elif file_type == "text/csv":
            return read_csv(file)

        elif file_type.endswith("spreadsheetml.sheet"):
            return read_xlsx(file)

        return None

    except Exception as e:
        print("Loader crash:", e)
        return None