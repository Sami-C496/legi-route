import { useState } from 'react'
import { Check, Copy } from 'lucide-react'
import type { Source } from './SourceCard'
import { SourceCard } from './SourceCard'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  streaming?: boolean
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* ignore */
    }
  }
  return (
    <button
      type="button"
      onClick={handleCopy}
      aria-label={copied ? 'Copié' : 'Copier la réponse'}
      title={copied ? 'Copié' : 'Copier'}
      className="inline-flex h-7 w-7 items-center justify-center rounded-md border border-rule bg-vellum text-slate transition-colors hover:border-marine hover:text-marine"
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
    </button>
  )
}

export function Message({ message }: { message: ChatMessage }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end animate-fade-in">
        <div className="max-w-[85%] rounded-lg rounded-br-sm bg-marine px-4 py-2.5 text-[15px] leading-relaxed text-vellum">
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <article className="animate-fade-in">
      {message.sources && message.sources.length > 0 && (
        <div className="mb-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
          {message.sources.map((s) => (
            <SourceCard key={s.article_number} source={s} />
          ))}
        </div>
      )}
      <div className="whitespace-pre-wrap font-serif text-[17px] leading-[1.65] text-ink">
        {message.content}
        {message.streaming && (
          <span className="ml-0.5 inline-block h-[1.1em] w-[2px] -mb-[2px] bg-ink align-middle animate-caret" />
        )}
      </div>
      {!message.streaming && message.content && (
        <div className="mt-2 flex justify-end">
          <CopyButton text={message.content} />
        </div>
      )}
    </article>
  )
}
