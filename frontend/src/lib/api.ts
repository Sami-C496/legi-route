import { readSSE } from './sse'
import type { Source } from '@/components/SourceCard'

export interface ChatTurn { role: 'user' | 'assistant'; content: string }

export interface StreamCallbacks {
  onIntent?: (intent: string) => void
  onSources?: (sources: Source[]) => void
  onToken?: (text: string) => void
  onDone?: (elapsed: number) => void
  onError?: (message: string) => void
}

export async function streamChat(
  prompt: string,
  history: ChatTurn[],
  cb: StreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, history, k: 3 }),
    signal,
  })
  if (!res.ok) {
    cb.onError?.(`Erreur ${res.status}`)
    return
  }
  for await (const { event, data } of readSSE(res)) {
    const d = data as any
    if (event === 'intent') cb.onIntent?.(d.intent)
    else if (event === 'sources') cb.onSources?.(d as Source[])
    else if (event === 'token') cb.onToken?.(d.text)
    else if (event === 'done') cb.onDone?.(d.elapsed)
    else if (event === 'error') cb.onError?.(d.message ?? 'Erreur')
  }
}
