export type Status = 'idle' | 'loading'

interface Props { status: Status }

export function StatusDot({ status }: Props) {
  const isLoading = status === 'loading'
  return (
    <div className="flex items-center gap-3 select-none" aria-live="polite">
      <div
        className="inline-flex items-center gap-2 rounded-full border border-rule bg-ink px-3 py-2 shadow-soft"
        role="img"
        aria-label={isLoading ? 'Recherche en cours' : 'Prêt'}
      >
        <span
          style={{ backgroundColor: isLoading ? '#C0392B' : '#3A3A3A' }}
          className={['inline-block h-4 w-4 rounded-full', isLoading ? 'animate-signal-pulse' : ''].join(' ')}
        />
        <span
          style={{ backgroundColor: '#3A3A3A' }}
          className="inline-block h-4 w-4 rounded-full"
        />
        <span
          style={{ backgroundColor: isLoading ? '#3A3A3A' : '#3F7D5C' }}
          className="inline-block h-4 w-4 rounded-full"
        />
      </div>
      <span className="text-[11px] uppercase tracking-[0.12em] text-slate">
        {isLoading ? 'Recherche' : 'Prêt'}
      </span>
    </div>
  )
}
