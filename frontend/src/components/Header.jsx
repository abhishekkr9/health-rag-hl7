import styles from './Header.module.css'

export default function Header({ connected }) {
  return (
    <header className={styles.header}>
      <div className={styles.brand}>
        <span className={styles.logotype}>FHIR</span>
        <span className={styles.separator}>·</span>
        <span className={styles.logotype}>RAG</span>
      </div>
      <div className={styles.meta}>
        <span className={styles.modelLabel}>llama-3.1-8b-instant</span>
        <span
          className={`${styles.pill} ${
            connected ? styles.pillConnected : styles.pillDisconnected
          }`}
        >
          <span className={styles.pillDot} aria-hidden="true" />
          {connected ? 'connected' : 'offline'}
        </span>
      </div>
    </header>
  )
}
