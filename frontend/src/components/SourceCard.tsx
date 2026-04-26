import { ExternalLink } from 'lucide-react'

export interface Source {
  article_number: string
  url: string
  excerpt: string
  score: number
}

export function SourceCard({ source }: { source: Source }) {
  return (
    <a
      href={source.url}
      target="_blank"
      rel="noreferrer"
      className="group block rounded-md border border-rule bg-vellum px-4 py-3 text-sm transition-colors hover:border-marine"
    >
      <div className="flex items-baseline justify-between gap-3">
        <span className="font-mono text-[13px] font-medium text-marine">
          Article {source.article_number}
        </span>
        <ExternalLink size={12} className="flex-shrink-0 text-slate group-hover:text-marine" />
      </div>
      <p className="mt-1.5 line-clamp-3 text-[13px] leading-relaxed text-slate">
        {source.excerpt}
      </p>
    </a>
  )
}
