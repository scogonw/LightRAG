import { expect, mock, test } from 'bun:test'
import { waitForDeletionJob, type DeletionJobStatus } from './deletionJob'

const running: DeletionJobStatus = { status: 'running', deleted_count: 0, failed_count: 0 }

test('waits for a confirmed deletion result', async () => {
  const completed: DeletionJobStatus = { status: 'completed', deleted_count: 2, failed_count: 0 }
  const getStatus = mock(async () => completed).mockResolvedValueOnce(running)

  await expect(waitForDeletionJob(getStatus, { pollIntervalMs: 0 })).resolves.toEqual(completed)
  expect(getStatus).toHaveBeenCalledTimes(2)
})

test('returns a partial failure as soon as the job reports it', async () => {
  const failed: DeletionJobStatus = { status: 'failed', deleted_count: 1, failed_count: 1 }
  const getStatus = mock(async () => failed)

  await expect(waitForDeletionJob(getStatus)).resolves.toEqual(failed)
  expect(getStatus).toHaveBeenCalledTimes(1)
})

test('returns no confirmation if the job is still running at the deadline', async () => {
  const getStatus = mock(async () => running)

  await expect(waitForDeletionJob(getStatus, { timeoutMs: 5 })).resolves.toBeNull()
})
