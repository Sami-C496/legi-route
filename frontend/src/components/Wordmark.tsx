import logoUrl from '/logo.svg'

export function Wordmark() {
  return (
    <div className="flex items-center gap-3">
      <img src={logoUrl} alt="" width={44} height={44} className="select-none" />
      <span className="wordmark text-[28px] leading-none text-ink">LégiRoute</span>
    </div>
  )
}
