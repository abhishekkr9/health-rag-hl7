import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble.jsx'
import styles from './ChatThread.module.css'

const EXAMPLES = [
  'What medications is Greenfelder433 currently taking?',
  'List all patients diagnosed with diabetes.',
  'What are the latest observations for Mante251?',
]

export default function ChatThread({ messages, loading, onExample }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  if (messages.length === 0) {
    return (
      <div className={styles.thread}>
        <div className={styles.empty}>
          <h1 className={styles.emptyHeading}>Clinical Intelligence</h1>
          <p className={styles.emptySubtext}>
            Query patient records in plain language. Answers are grounded in
            FHIR data only — no hallucination.
          </p>
          <div className={styles.exampleBlock}>
            <span className={styles.exampleEyebrow}>Try asking</span>
            <ul className={styles.exampleList}>
              {EXAMPLES.map((q, i) => (
                <li key={i}>
                  <button
                    className={styles.exampleBtn}
                    onClick={() => onExample(q)}
                  >
                    {q}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.thread}>
      <div className={styles.messages}>
        {messages.map((msg, i) => (
          <MessageBubble
            key={i}
            message={msg}
            isLast={i === messages.length - 1}
          />
        ))}
        {loading && (
          <div className={styles.thinking} aria-label="Retrieving answer">
            <span className={styles.dot} />
            <span className={styles.dot} />
            <span className={styles.dot} />
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
