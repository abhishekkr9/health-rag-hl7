import styles from './Sidebar.module.css'

export default function Sidebar({ patients }) {
  return (
    <aside className={styles.sidebar} aria-label="Session details">
      <dl className={styles.facts}>
        <div className={styles.fact}>
          <dt className={styles.factLabel}>Model</dt>
          <dd className={styles.factValue}>llama-3.1-8b-instant</dd>
        </div>
        <div className={styles.fact}>
          <dt className={styles.factLabel}>Embeddings</dt>
          <dd className={styles.factValue}>PubMedBERT</dd>
        </div>
        <div className={styles.fact}>
          <dt className={styles.factLabel}>Search</dt>
          <dd className={styles.factValue}>Hybrid · α=0.5</dd>
        </div>
        <div className={styles.fact}>
          <dt className={styles.factLabel}>Vector store</dt>
          <dd className={styles.factValue}>Weaviate</dd>
        </div>
        <div className={styles.fact}>
          <dt className={styles.factLabel}>Records</dt>
          <dd className={styles.factValue}>FHIR R4</dd>
        </div>
      </dl>

      {patients.length > 0 && (
        <section className={styles.contextSection}>
          <span className={styles.contextLabel}>In context</span>
          <ul className={styles.chips}>
            {patients.map((name, i) => (
              <li key={i} className={styles.chip} title={name}>
                {name}
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className={styles.spacer} aria-hidden="true" />

      <div className={styles.footer}>
        <span className={styles.footerNote}>FHIR·RAG</span>
      </div>
    </aside>
  )
}
