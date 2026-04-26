import { useEffect, useRef, useState } from 'react'
import { Wordmark } from './components/Wordmark'
import { StatusDot, type Status } from './components/StatusDot'
import { ChatInput } from './components/ChatInput'
import { Message, type ChatMessage } from './components/Message'
import { streamChat, type ChatTurn } from './lib/api'

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [status, setStatus] = useState<Status>('idle')
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  async function handleSubmit(prompt: string) {
    setStatus('loading')
    const history: ChatTurn[] = messages
      .filter((m) => !m.streaming)
      .map((m) => ({ role: m.role, content: m.content }))

    setMessages((prev) => [
      ...prev,
      { role: 'user', content: prompt },
      { role: 'assistant', content: '', streaming: true },
    ])

    const updateAssistant = (patch: Partial<ChatMessage>) =>
      setMessages((prev) => {
        const next = [...prev]
        const last = next[next.length - 1]
        next[next.length - 1] = { ...last, ...patch }
        return next
      })

    let buffer = ''
    let sources: ChatMessage['sources'] = []

    await streamChat(prompt, history, {
      onSources: (s) => {
        sources = s
        updateAssistant({ sources: s })
      },
      onToken: (t) => {
        buffer += t
        updateAssistant({ content: buffer, sources })
      },
      onError: (msg) => {
        updateAssistant({ content: msg, streaming: false })
      },
      onDone: () => {
        updateAssistant({ streaming: false })
      },
    }).catch((e) => {
      updateAssistant({
        content: `Erreur de connexion : ${e?.message ?? e}`,
        streaming: false,
      })
    })

    setStatus('idle')
  }

  return (
    <div className="flex h-screen flex-col bg-paper">
      <header className="border-b border-rule bg-paper">
        <div className="mx-auto flex w-full max-w-column items-center justify-between px-6 py-5">
          <Wordmark />
          <StatusDot status={status} />
        </div>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-column px-6 py-10">
          {messages.length === 0 ? (
            <p className="font-serif text-[18px] leading-relaxed text-slate">
              Je réponds à vos questions sur le Code de la Route avec précision, en citant systématiquement les articles officiels.
            </p>
          ) : (
            <div className="flex flex-col gap-7">
              {messages.map((m, i) => (
                <Message key={i} message={m} />
              ))}
            </div>
          )}
        </div>
      </div>

      <footer className="border-t border-rule bg-paper">
        <div className="mx-auto w-full max-w-column px-6 py-4">
          <ChatInput onSubmit={handleSubmit} disabled={status === 'loading'} />
          <p className="mt-2 text-center text-[11px] text-slate">
            Outil informatif. Ne remplace pas le conseil d'un professionnel du droit.
          </p>
        </div>
      </footer>
    </div>
  )
}
