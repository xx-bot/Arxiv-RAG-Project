# full pdf text cleaning
import re

def clean_paper(text):
    # remove citations like [1], [2-5], ...
    text = re.sub(r"\[\d+(?:-\s*\d+)*\]", '', text)

    # remove author citation, like K. et al.
    text = re.sub(r"[a-zA-Z]\.\s*\w*\s*et al.", '', text)

    # remove author citation, like (author, year)
    text = re.sub(r"\(.+,\s*\d{4}\)", '', text)

    # remove extra whitespace after cleaning
    text = re.sub(r'\s+', ' ', text).strip()

    return text
