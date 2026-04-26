import type { Source } from './SourceCard'
import { SourceCard } from './SourceCard'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  streaming?: boolean
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
    </article>
  )
}
