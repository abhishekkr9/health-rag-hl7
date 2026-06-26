/**
 * Extracts patient names from FHIR RAG responses.
 *
 * The FHIR parsers emit lines like:
 *   "Patient: Fletcher87 Greenfelder433"
 *   "Patient: Zane918 Mante251"
 *
 * This function collects the names that follow "Patient:" so the sidebar
 * can show which patients are referenced in the current conversation.
 */
export function extractPatients(text) {
  const pattern = /Patient:\s+([^\n,]+)/g
  const names = []
  let match

  while ((match = pattern.exec(text)) !== null) {
    const name = match[1].trim()
    // Guard against runaway matches
    if (name && name.length > 0 && name.length < 80) {
      names.push(name)
    }
  }

  return [...new Set(names)]
}
