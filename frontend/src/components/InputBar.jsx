import { useState, useRef } from 'react'
import styles from './InputBar.module.css'

export default function InputBar({ onSend, disabled }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  const submit = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
    textareaRef.current?.focus()
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const handleChange = (e) => {
    setValue(e.target.value)
    // Auto-grow textarea up to ~120px
    const ta = e.target
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 120)}px`
  }

  return (
    <div className={styles.bar}>
      <div className={styles.inner}>
        <textarea
          ref={textareaRef}
          className={styles.input}
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder="Ask about patients, diagnoses, medications, observations…"
          rows={1}
          disabled={disabled}
          aria-label="Ask a question about patient records"
        />
        <button
          className={styles.send}
          onClick={submit}
          disabled={disabled || !value.trim()}
          aria-label="Send question"
        >
          <ArrowIcon />
        </button>
      </div>
      <p className={styles.hint}>
        Enter to send · Shift+Enter for new line
      </p>
    </div>
  )
}

function ArrowIcon() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 15 15"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M1.5 7.5h12M8.5 2.5l5 5-5 5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
