import { useCallback, useEffect, useState } from 'react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import { useSettingsStore } from '@/stores/settings'
import { getGraphOrgIds } from '@/api/lightrag'

/** Sentinel for "no tenant selected" — Radix Select cannot hold an empty value. */
const ALL_ORGS = '__all__'

/**
 * Tenant selector for the graph viewer.
 *
 * Without a selection the viewer calls the cross-org `/graphs` and shows every
 * tenant's nodes at once, which is rarely what you want and mixes entity
 * descriptions from unrelated orgs. Selecting a tenant switches the viewer to
 * `/graphs/by_org` and scopes `X-Org-Id` for the rest of the UI.
 *
 * This filters the view; it does not restrict access. Any authenticated user
 * can select any tenant, because the auth layer carries no org binding.
 */
const GraphOrgSelector = () => {
  const orgId = useSettingsStore.use.orgId()
  const [orgIds, setOrgIds] = useState<string[]>([])
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let cancelled = false
    getGraphOrgIds()
      .then((ids) => {
        if (!cancelled) {
          setOrgIds(ids)
          setFailed(false)
        }
      })
      .catch((error) => {
        console.error('Failed to load graph org IDs:', error)
        if (!cancelled) setFailed(true)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const onChange = useCallback((value: string) => {
    useSettingsStore.getState().setOrgId(value === ALL_ORGS ? null : value)
  }, [])

  // Backends that cannot enumerate orgs (non-OpenSearch graph storage) return
  // an empty list. Hiding the control is better than offering an empty one that
  // looks broken.
  if (failed || orgIds.length === 0) return null

  return (
    <Select value={orgId ?? ALL_ORGS} onValueChange={onChange}>
      <SelectTrigger className="bg-background/60 h-9 w-[190px] backdrop-blur-lg">
        <SelectValue placeholder="All tenants" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={ALL_ORGS}>All tenants</SelectItem>
        {orgIds.map((id) => (
          <SelectItem key={id} value={id}>
            {id}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

export default GraphOrgSelector
