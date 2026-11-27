export type DeletionJobStatus = {
  status: 'running' | 'completed' | 'failed'
  deleted_count: number
  failed_count: number
}

export const waitForDeletionJob = async (
  getStatus: () => Promise<DeletionJobStatus>,
  { timeoutMs = 10 * 60_000, pollIntervalMs = 2000 } = {}
): Promise<DeletionJobStatus | null> => {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const result = await getStatus()
    if (result.status === 'completed' || result.status === 'failed') {
      return result
    }
    const delay = Math.min(pollIntervalMs, Math.max(0, deadline - Date.now()))
    await new Promise(resolve => setTimeout(resolve, delay))
  }
  return null
}
