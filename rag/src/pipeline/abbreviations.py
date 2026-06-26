"""Medical abbreviation expansion — static lookup for common clinical shorthand."""

import re

# Common unambiguous medical abbreviations → full terms
ABBREVIATIONS: dict[str, str] = {
    # Conditions
    "HTN":   "hypertension",
    "DM":    "diabetes mellitus",
    "DM1":   "type 1 diabetes mellitus",
    "DM2":   "type 2 diabetes mellitus",
    "T2DM":  "type 2 diabetes mellitus",
    "T1DM":  "type 1 diabetes mellitus",
    "CAD":   "coronary artery disease",
    "CHF":   "congestive heart failure",
    "COPD":  "chronic obstructive pulmonary disease",
    "CKD":   "chronic kidney disease",
    "ESRD":  "end-stage renal disease",
    "CVA":   "cerebrovascular accident stroke",
    "TIA":   "transient ischemic attack",
    "MI":    "myocardial infarction heart attack",
    "PE":    "pulmonary embolism",
    "DVT":   "deep vein thrombosis",
    "GERD":  "gastroesophageal reflux disease",
    "IBD":   "inflammatory bowel disease",
    "IBS":   "irritable bowel syndrome",
    "RA":    "rheumatoid arthritis",
    "OA":    "osteoarthritis",
    "OSA":   "obstructive sleep apnea",
    "PTSD":  "post-traumatic stress disorder",
    "MDD":   "major depressive disorder",
    "GAD":   "generalized anxiety disorder",
    "BPH":   "benign prostatic hyperplasia",
    "UTI":   "urinary tract infection",
    "URI":   "upper respiratory infection",
    "URTI":  "upper respiratory tract infection",
    "AKI":   "acute kidney injury",
    "AFib":  "atrial fibrillation",
    "AFIB":  "atrial fibrillation",
    "A-fib": "atrial fibrillation",
    "HF":    "heart failure",

    # Medications / treatments
    "BP":    "blood pressure",
    "HR":    "heart rate",
    "RR":    "respiratory rate",
    "SpO2":  "oxygen saturation",
    "BMI":   "body mass index",
    "HbA1c": "hemoglobin A1c glycated hemoglobin",
    "LDL":   "low-density lipoprotein cholesterol",
    "HDL":   "high-density lipoprotein cholesterol",
    "TSH":   "thyroid-stimulating hormone",
    "CBC":   "complete blood count",
    "BMP":   "basic metabolic panel",
    "CMP":   "comprehensive metabolic panel",
    "EKG":   "electrocardiogram",
    "ECG":   "electrocardiogram",
    "MRI":   "magnetic resonance imaging",
    "CT":    "computed tomography scan",
    "CXR":   "chest X-ray",
    "US":    "ultrasound",

    # Procedures / encounters
    "ED":    "emergency department",
    "ER":    "emergency room",
    "ICU":   "intensive care unit",
    "OR":    "operating room",
    "PCP":   "primary care physician",
    "OB":    "obstetrics",
    "GYN":   "gynecology",
    "OBGYN": "obstetrics and gynecology",
    "ENT":   "ear nose and throat",

    # Misc clinical
    "Hx":    "history",
    "HX":    "history",
    "Dx":    "diagnosis",
    "DX":    "diagnosis",
    "Rx":    "prescription medication",
    "RX":    "prescription medication",
    "Sx":    "symptoms",
    "SX":    "symptoms",
    "Tx":    "treatment",
    "TX":    "treatment",
    "SOB":   "shortness of breath",
    "DOE":   "dyspnea on exertion shortness of breath on exertion",
    "CP":    "chest pain",
    "HA":    "headache",
    "N/V":   "nausea and vomiting",
    "N/V/D": "nausea vomiting and diarrhea",
    "WNL":   "within normal limits",
    "PMH":   "past medical history",
    "FH":    "family history",
    "SH":    "social history",
    "NKDA":  "no known drug allergies",
    "NKA":   "no known allergies",
    "yo":    "years old",
    "y/o":   "years old",
    "F/U":   "follow-up",
    "f/u":   "follow-up",
    "w/":    "with",
    "s/p":   "status post",
    "S/P":   "status post",
    "h/o":   "history of",
    "H/O":   "history of",
    "c/o":   "complains of",
    "C/O":   "complains of",
}

# Build a regex that matches whole words (case-sensitive for abbreviations)
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(ABBREVIATIONS, key=len, reverse=True)) + r")\b"
)


def expand(text: str) -> str:
    """
    Expand medical abbreviations in text.

    Only replaces whole-word matches to avoid false positives.
    Original abbreviation is kept alongside the expansion:
        "HTN" → "HTN (hypertension)"
    """
    def _replace(m: re.Match) -> str:
        abbr = m.group(0)
        full = ABBREVIATIONS.get(abbr, "")
        if not full:
            return abbr
        return f"{abbr} ({full})"

    return _PATTERN.sub(_replace, text)
