import { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import Button from '@/components/ui/Button'
import { getDocumentHealth, type DocumentHealthReport } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'

interface DocumentHealthDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

/** What each class means, shown inline so the numbers are actionable. */
const CLASS_HELP: Record<string, string> = {
  healthy: 'Owns its chunks, and they carry tenant metadata.',
  orphan: 'Owns chunks that nothing has cascaded onto — usually no live upstream row. Verify before deleting.',
  shared: 'Owns no chunks, but every chunk it lists is alive: a sibling owns them. Its resource may still be the only anchor for one audience, so do not delete blindly.',
  inert: 'Owns no chunks and the ones it lists are gone. Delete if the resource is dead, re-embed if it is live.',
  empty: 'Declares no chunks at all.'
}

const CLASS_ORDER = ['healthy', 'orphan', 'shared', 'inert', 'empty']

export default function DocumentHealthDialog({
  open,
  onOpenChange
}: DocumentHealthDialogProps) {
  const [report, setReport] = useState<DocumentHealthReport | null>(null)
  const [loading, setLoading] = useState(false)

  const run = useCallback(async () => {
    setLoading(true)
    try {
      setReport(await getDocumentHealth())
    } catch (error) {
      toast.error(`Health check failed: ${errorMessage(error)}`)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (open && !report && !loading) {
      run()
    }
  }, [open, report, loading, run])

  const integrityRows = report
    ? [
        ['KV / vector access mismatches', report.integrity.access_mismatches],
        ['Chunks with no metadata', report.integrity.chunks_without_metadata],
        ['Cross-org entries', report.integrity.cross_org_entries]
      ]
    : []

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85vh] max-w-3xl overflow-auto">
        <DialogHeader>
          <DialogTitle>Document health</DialogTitle>
          <DialogDescription>
            Chunks are content-addressed, so one chunk record can belong to several
            documents. This reports which documents actually own their content, and
            whether the storage invariants still hold.
          </DialogDescription>
        </DialogHeader>

        {loading && <div className="py-8 text-center text-sm">Scanning…</div>}

        {report && !loading && (
          <div className="flex flex-col gap-4 text-sm">
            <div className="text-muted-foreground">
              {report.documents} documents · {report.chunks.kv} KV chunks ·{' '}
              {report.chunks.vector} vector chunks
            </div>

            <div className="flex flex-col gap-1">
              {CLASS_ORDER.map((klass) => {
                const stats = report.classes[klass]
                if (!stats) return null
                return (
                  <div key={klass} className="flex items-baseline gap-2">
                    <span className="w-20 font-mono">{klass}</span>
                    <span className="w-28 tabular-nums">
                      {stats.documents} docs
                    </span>
                    <span className="w-32 tabular-nums text-muted-foreground">
                      {stats.chunks_owned} chunks
                    </span>
                    <span className="text-muted-foreground flex-1 text-xs">
                      {CLASS_HELP[klass]}
                    </span>
                  </div>
                )
              })}
            </div>

            <div className="flex flex-col gap-1 border-t pt-3">
              {integrityRows.map(([label, value]) => (
                <div key={String(label)} className="flex items-baseline gap-2">
                  <span className="w-56">{label}</span>
                  <span
                    className={
                      Number(value) > 0
                        ? 'font-semibold text-red-600 tabular-nums'
                        : 'tabular-nums'
                    }
                  >
                    {value}
                  </span>
                </div>
              ))}
              <div className="text-muted-foreground text-xs">
                All three should read zero.
              </div>
            </div>

            {report.actionable.length > 0 && (
              <div className="flex flex-col gap-2 border-t pt-3">
                <div className="font-medium">
                  Needs adjudicating upstream ({report.actionable.length})
                </div>
                <div className="text-muted-foreground text-xs">
                  LightRAG knows a document&apos;s shape; only the system of record
                  knows whether its resource is still live. Check these
                  resource&nbsp;IDs there before deleting or re-embedding.
                </div>
                <div className="max-h-64 overflow-auto rounded border">
                  <table className="w-full text-xs">
                    <thead className="bg-muted/50 sticky top-0">
                      <tr>
                        <th className="p-1 text-left">document</th>
                        <th className="p-1 text-left">class</th>
                        <th className="p-1 text-right">owned</th>
                        <th className="p-1 text-left">access</th>
                        <th className="p-1 text-left">resource_id</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.actionable.map((row) => (
                        <tr key={row.doc_id} className="border-t">
                          <td className="p-1" title={row.doc_id}>
                            {row.file_path ?? row.doc_id}
                          </td>
                          <td className="p-1 font-mono">{row.klass}</td>
                          <td className="p-1 text-right tabular-nums">{row.owned}</td>
                          <td className="p-1">{row.levels.join(', ')}</td>
                          <td className="p-1 font-mono">{row.resource_id ?? '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="flex justify-end gap-2">
          <Button variant="outline" size="sm" onClick={run} disabled={loading}>
            {loading ? 'Scanning…' : 'Re-run'}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
