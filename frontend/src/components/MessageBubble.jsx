import { useState, useEffect, useRef } from 'react'
import styles from './MessageBubble.module.css'

// Characters revealed per tick — 4 chars at 12ms ≈ ~330 chars/sec.
// Fast enough to feel responsive, slow enough to read as a printout.
const CHARS_PER_TICK = 4
const TICK_MS = 12

export default function MessageBubble({ message }) {
  const { role, content, streaming } = message

  const [displayed, setDisplayed] = useState(streaming ? '' : content)
  const [done, setDone] = useState(!streaming)
  const indexRef = useRef(streaming ? 0 : content.length)

  useEffect(() => {
    if (!streaming || done) return

    const tick = () => {
      indexRef.current = Math.min(indexRef.current + CHARS_PER_TICK, content.length)
      setDisplayed(content.slice(0, indexRef.current))
      if (indexRef.current >= content.length) {
        setDone(true)
      }
    }

    const id = setInterval(tick, TICK_MS)
    return () => clearInterval(id)
  }, [streaming, content, done])

  if (role === 'user') {
    return (
      <div className={styles.userRow}>
        <div className={styles.userBubble}>
          <p className={styles.userText}>{content}</p>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.assistantRow}>
      <div className={styles.assistantHeader}>
        <span className={styles.assistantLabel}>Clinical Record</span>
        <span className={styles.rule} aria-hidden="true" />
      </div>
      <pre className={styles.assistantText}>
        {displayed}
        {!done && <span className={styles.cursor} aria-hidden="true" />}
      </pre>
    </div>
  )
}
