import Header from './components/Header.jsx'
import Sidebar from './components/Sidebar.jsx'
import ChatThread from './components/ChatThread.jsx'
import InputBar from './components/InputBar.jsx'
import { useChat } from './hooks/useChat.js'
import styles from './App.module.css'

export default function App() {
  const { messages, loading, connected, patients, sendMessage } = useChat()

  return (
    <div className={styles.app}>
      <Header connected={connected} />
      <div className={styles.body}>
        <Sidebar patients={patients} />
        <div className={styles.chatArea}>
          <ChatThread messages={messages} loading={loading} onExample={sendMessage} />
          <InputBar onSend={sendMessage} disabled={loading} />
        </div>
      </div>
    </div>
  )
}
