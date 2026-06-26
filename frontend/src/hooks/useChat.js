import { useState, useEffect, useCallback } from 'react'
import { sendChat, checkHealth } from '../utils/api.js'
import { extractPatients } from '../utils/parsePatients.js'

const THREAD_KEY = 'fhir_rag_thread_id'

function getOrCreateThreadId() {
  let id = sessionStorage.getItem(THREAD_KEY)
  if (!id) {
    id = crypto.randomUUID()
    sessionStorage.setItem(THREAD_KEY, id)
  }
  return id
}

export function useChat() {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [connected, setConnected] = useState(false)
  const [patients, setPatients] = useState([])

  // Thread ID is stable for the browser session, maps to LangGraph checkpointer
  const threadId = getOrCreateThreadId()

  // Check backend health on mount
  useEffect(() => {
    checkHealth()
      .then(() => setConnected(true))
      .catch(() => setConnected(false))
  }, [])

  const sendMessage = useCallback(
    async (question) => {
      setMessages((prev) => [...prev, { role: 'user', content: question }])
      setLoading(true)

      try {
        const { answer } = await sendChat({ question, thread_id: threadId })

        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: answer, streaming: true },
        ])

        // Extract any patient names mentioned in the response for the sidebar
        const found = extractPatients(answer)
        if (found.length > 0) {
          setPatients((prev) => [...new Set([...prev, ...found])])
        }
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content:
              "Unable to retrieve an answer. Check that the backend server is running and the OPENROUTER_API_KEY is set.",
            streaming: false,
          },
        ])
      } finally {
        setLoading(false)
      }
    },
    [threadId],
  )

  return { messages, loading, connected, patients, sendMessage }
}
