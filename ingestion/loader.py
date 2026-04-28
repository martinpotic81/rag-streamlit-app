
import pdfplumber
import docx
import pandas as pd

def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def read_xlsx(file):
    df = pd.read_excel(file)
    return df.to_string()

def load_file(file):
    if file.type == "application/pdf":
        return read_pdf(file)
    elif file.type.endswith("wordprocessingml.document"):
        return read_docx(file)
    elif file.type == "text/csv":
        return read_csv(file)
    elif file.type.endswith("spreadsheetml.sheet"):
        return read_xlsx(file)
    else:
        return ""