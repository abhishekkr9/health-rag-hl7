"""FHIR resource parsers — convert each resource type to a plain-text string."""

import base64


def _code_text(obj: dict) -> str:
    if not obj:
        return ""
    if obj.get("text"):
        return obj["text"]
    codings = obj.get("coding", [])
    if codings:
        return codings[0].get("display") or codings[0].get("code", "")
    return ""


def _date(val: str) -> str:
    return val[:10] if val else ""


def _ext_value(extensions: list, url_fragment: str) -> str:
    """Extract valueString or valueCoding.display from a named extension."""
    for ext in extensions:
        if url_fragment in ext.get("url", ""):
            inner = ext.get("extension", [])
            if inner:
                return (
                    inner[0].get("valueString")
                    or inner[0].get("valueCoding", {}).get("display", "")
                    or ""
                )
            return ext.get("valueString") or ext.get("valueCode", "")
    return ""


def parse_patient(r: dict) -> str:
    name       = r.get("name", [{}])[0]
    full       = " ".join(name.get("given", []) + [name.get("family", "")]).strip()
    gender     = r.get("gender", "unknown")
    dob        = _date(r.get("birthDate", ""))
    patient_id = r.get("id", "")
    exts       = r.get("extension", [])

    addr       = r.get("address", [{}])[0]
    addr_line  = ", ".join(addr.get("line", []))
    city_state = f"{addr.get('city', '')}, {addr.get('state', '')} {addr.get('postalCode', '')}".strip(", ")

    race       = _ext_value(exts, "us-core-race")
    ethnicity  = _ext_value(exts, "us-core-ethnicity")
    birth_sex  = _ext_value(exts, "us-core-birthsex")
    birth_place_ext = next((e for e in exts if "birthPlace" in e.get("url", "")), {})
    birth_place = ""
    if birth_place_ext:
        bp = birth_place_ext.get("valueAddress", {})
        birth_place = f"{bp.get('city', '')}, {bp.get('state', '')}".strip(", ")

    marital    = _code_text(r.get("maritalStatus", {}))
    telecom    = next((t["value"] for t in r.get("telecom", []) if t.get("system") == "phone"), "")
    deceased   = _date(r.get("deceasedDateTime", "")) or (
        "deceased" if r.get("deceasedBoolean") else ""
    )

    lines = [f"Patient: {full}", f"  ID: {patient_id}  DOB: {dob}  Gender: {gender}"]
    if birth_sex:
        lines.append(f"  Birth Sex: {birth_sex}")
    if addr_line or city_state:
        lines.append(f"  Address: {addr_line}, {city_state}".strip(", "))
    if birth_place:
        lines.append(f"  Birth Place: {birth_place}")
    if race:
        lines.append(f"  Race: {race}")
    if ethnicity:
        lines.append(f"  Ethnicity: {ethnicity}")
    if marital:
        lines.append(f"  Marital Status: {marital}")
    if telecom:
        lines.append(f"  Phone: {telecom}")
    if deceased:
        lines.append(f"  Deceased: {deceased}")
    return "\n".join(lines)


def parse_condition(r: dict) -> str:
    code     = _code_text(r.get("code", {}))
    status   = _code_text(r.get("clinicalStatus", {}))
    severity = _code_text(r.get("severity", {}))
    onset    = _date(r.get("onsetDateTime", ""))
    abated   = _date(r.get("abatementDateTime", ""))
    notes    = " | ".join(n.get("text", "") for n in r.get("note", []) if n.get("text"))
    parts    = [f"Condition: {code}  Status: {status}  Onset: {onset}"]
    if abated:
        parts.append(f"  Resolved: {abated}")
    if severity:
        parts.append(f"  Severity: {severity}")
    if notes:
        parts.append(f"  Notes: {notes}")
    return "\n".join(parts)


def parse_observation(r: dict) -> str:
    code  = _code_text(r.get("code", {}))
    vq    = r.get("valueQuantity", {})
    if vq:
        value = f"{vq.get('value', '')} {vq.get('unit', '')}".strip()
    else:
        value = _code_text(r.get("valueCodeableConcept", {})) or r.get("valueString", "")
    date  = _date(r.get("effectiveDateTime", ""))
    interp = _code_text((r.get("interpretation") or [{}])[0])
    ref_range = ""
    rr = (r.get("referenceRange") or [{}])[0]
    low  = rr.get("low", {}).get("value", "")
    high = rr.get("high", {}).get("value", "")
    unit = rr.get("low", rr.get("high", {})).get("unit", "")
    if low or high:
        ref_range = f"{low}-{high} {unit}".strip()
    parts = [f"Observation: {code} = {value}  ({date})"]
    if interp:
        parts.append(f"  Interpretation: {interp}")
    if ref_range:
        parts.append(f"  Reference Range: {ref_range}")
    return "\n".join(parts)


def parse_procedure(r: dict) -> str:
    code      = _code_text(r.get("code", {}))
    performed = _date(
        r.get("performedDateTime")
        or (r.get("performedPeriod") or {}).get("start", "")
    )
    status    = r.get("status", "")
    reason    = _code_text((r.get("reasonCode") or [{}])[0])
    parts     = [f"Procedure: {code}  Date: {performed}  Status: {status}"]
    if reason:
        parts.append(f"  Reason: {reason}")
    return "\n".join(parts)


def parse_medication_request(r: dict) -> str:
    med    = _code_text(r.get("medicationCodeableConcept", {}))
    date   = _date(r.get("authoredOn", ""))
    status = r.get("status", "")
    dosage = (r.get("dosageInstruction") or [{}])[0].get("text", "")
    parts  = [f"Medication: {med}  Prescribed: {date}  Status: {status}"]
    if dosage:
        parts.append(f"  Dosage: {dosage}")
    return "\n".join(parts)


def parse_encounter(r: dict) -> str:
    etype    = _code_text((r.get("type") or [{}])[0])
    period   = r.get("period") or {}
    start    = _date(period.get("start", ""))
    end      = _date(period.get("end", ""))
    status   = r.get("status", "")
    reason   = _code_text((r.get("reasonCode") or [{}])[0])
    provider = (r.get("serviceProvider") or {}).get("display", "")
    doctor   = (r.get("participant") or [{}])[0].get("individual", {}).get("display", "")
    parts    = [f"Encounter: {etype}  Status: {status}  Date: {start}" + (f" to {end}" if end else "")]
    if reason:
        parts.append(f"  Reason: {reason}")
    if provider:
        parts.append(f"  Provider: {provider}")
    if doctor:
        parts.append(f"  Clinician: {doctor}")
    return "\n".join(parts)


def parse_allergy(r: dict) -> str:
    substance  = _code_text(r.get("code", {}))
    reaction   = _code_text(((r.get("reaction") or [{}])[0].get("manifestation") or [{}])[0])
    severity   = (r.get("reaction") or [{}])[0].get("severity", "")
    criticality = r.get("criticality", "")
    date       = _date(r.get("recordedDate", ""))
    parts      = [f"Allergy: {substance}  Reaction: {reaction}  Recorded: {date}"]
    if severity:
        parts.append(f"  Severity: {severity}")
    if criticality:
        parts.append(f"  Criticality: {criticality}")
    return "\n".join(parts)


def parse_immunization(r: dict) -> str:
    vaccine = _code_text(r.get("vaccineCode", {}))
    date    = _date(r.get("occurrenceDateTime", ""))
    status  = r.get("status", "")
    return f"Immunization: {vaccine}  Date: {date}  Status: {status}"


def parse_diagnostic_report(r: dict) -> str:
    code       = _code_text(r.get("code", {}))
    date       = _date(r.get("effectiveDateTime", ""))
    status     = r.get("status", "")
    conclusion = r.get("conclusion", "")
    parts      = [f"DiagnosticReport: {code}  Date: {date}  Status: {status}"]
    if conclusion:
        parts.append(f"  Conclusion: {conclusion}")
    return "\n".join(parts)


def parse_document_reference(r: dict) -> str:
    doc_type = _code_text(r.get("type", {}))
    date     = _date(r.get("date", ""))
    status   = r.get("status", "")
    author   = (r.get("author") or [{}])[0].get("display", "")
    parts    = [f"ClinicalNote: {doc_type}  Date: {date}  Status: {status}"]
    if author:
        parts.append(f"  Author: {author}")
    for content in r.get("content", []):
        att = content.get("attachment", {})
        data = att.get("data", "")
        if data:
            try:
                text = base64.b64decode(data).decode("utf-8", errors="replace").strip()
                if text:
                    parts.append(f"  Note:\n{text}")
            except Exception:
                pass
    return "\n".join(parts)


def parse_organization(r: dict) -> str:
    name     = r.get("name", "Unknown")
    active   = "active" if r.get("active") else "inactive"
    org_type = _code_text(r.get("type", [{}])[0]) if r.get("type") else ""
    address  = r.get("address", [{}])[0]
    addr_parts = [
        ", ".join(address.get("line", [])),
        address.get("city", ""),
        address.get("state", ""),
        address.get("postalCode", ""),
        address.get("country", ""),
    ]
    addr_str = ", ".join(p for p in addr_parts if p)
    telecom  = next((t["value"] for t in r.get("telecom", []) if t.get("system") == "phone"), "")
    parts    = [f"Organization: {name}  Type: {org_type}  Status: {active}"]
    if addr_str:
        parts.append(f"  Address: {addr_str}")
    if telecom:
        parts.append(f"  Phone: {telecom}")
    return "\n".join(parts)


def parse_location(r: dict) -> str:
    name    = r.get("name", "Unknown")
    status  = r.get("status", "")
    address = r.get("address", {})
    addr_parts = [
        ", ".join(address.get("line", [])),
        address.get("city", ""),
        address.get("state", ""),
        address.get("postalCode", ""),
    ]
    addr_str = ", ".join(p for p in addr_parts if p)
    telecom  = next((t["value"] for t in r.get("telecom", []) if t.get("system") == "phone"), "")
    managing = r.get("managingOrganization", {}).get("display", "")
    parts    = [f"Location: {name}  Status: {status}"]
    if addr_str:
        parts.append(f"  Address: {addr_str}")
    if telecom:
        parts.append(f"  Phone: {telecom}")
    if managing:
        parts.append(f"  Managed by: {managing}")
    return "\n".join(parts)


PARSERS: dict[str, callable] = {
    "Patient":            parse_patient,
    "Condition":          parse_condition,
    "Observation":        parse_observation,
    "Procedure":          parse_procedure,
    "MedicationRequest":  parse_medication_request,
    "Encounter":          parse_encounter,
    "AllergyIntolerance": parse_allergy,
    "Immunization":       parse_immunization,
    "DiagnosticReport":   parse_diagnostic_report,
    "DocumentReference":  parse_document_reference,
    "Organization":       parse_organization,
    "Location":           parse_location,
}
