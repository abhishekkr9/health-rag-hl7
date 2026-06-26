const BASE = '/api'

export async function checkHealth() {
  const res = await fetch(`${BASE}/health`, { method: 'GET' })
  if (!res.ok) throw new Error('Health check failed')
  return res.json()
}

export async function sendChat({ question, thread_id }) {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, thread_id }),
  })
  if (!res.ok) {
    const message = await res.text().catch(() => 'Request failed')
    throw new Error(message)
  }
  return res.json()
}
