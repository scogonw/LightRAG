import { afterEach, beforeAll, describe, expect, test } from 'bun:test'

type DocumentsRequest = {
  status_filter?: 'pending' | 'processing' | 'preprocessed' | 'processed' | 'failed' | null
  page: number
  page_size: number
  sort_field: 'created_at' | 'updated_at' | 'id' | 'file_path'
  sort_direction: 'asc' | 'desc'
  file_path_filter?: string | null
}

type LightragApiModule = typeof import('./lightrag')

const storageMock = () => {
  const data = new Map<string, string>()

  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value)
    },
    removeItem: (key: string) => {
      data.delete(key)
    },
    clear: () => {
      data.clear()
    }
  }
}

let apiModule: LightragApiModule

beforeAll(async () => {
  Object.defineProperty(globalThis, 'localStorage', {
    value: storageMock(),
    configurable: true
  })
  Object.defineProperty(globalThis, 'sessionStorage', {
    value: storageMock(),
    configurable: true
  })

  apiModule = await import('./lightrag')
})

afterEach(() => {
  apiModule.__resetPaginatedDocumentRequestsForTests()
})

describe('getDocumentsPaginated', () => {
  test('issues a fresh request after aborting a timed-out in-flight request', async () => {
    const request: DocumentsRequest = {
      status_filter: null,
      page: 1,
      page_size: 20,
      sort_field: 'updated_at',
      sort_direction: 'desc'
    }

    let callCount = 0
    const resolvers: Array<(value: any) => void> = []

    apiModule.__setPaginatedDocumentsPostForTests((_request, controller) => {
      callCount += 1

      return new Promise((resolve, reject) => {
        resolvers.push(resolve)
        controller.signal.addEventListener(
          'abort',
          () => reject(new DOMException('Aborted', 'AbortError')),
          { once: true }
        )
      })
    })

    const firstRequest = apiModule.getDocumentsPaginated(request)
    const secondRequest = apiModule.getDocumentsPaginated(request)

    expect(callCount).toBe(1)

    apiModule.abortDocumentsPaginated(request)
    const [firstResult, secondResult] = await Promise.allSettled([
      firstRequest,
      secondRequest
    ])
    expect(firstResult.status).toBe('rejected')
    expect(secondResult.status).toBe('rejected')

    const thirdRequest = apiModule.getDocumentsPaginated(request)
    expect(callCount).toBe(2)

    resolvers[1]({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })

    await expect(thirdRequest).resolves.toEqual({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })
  })

  test('times out hanging requests and allows a fresh retry', async () => {
    const request: DocumentsRequest = {
      status_filter: null,
      page: 1,
      page_size: 20,
      sort_field: 'updated_at',
      sort_direction: 'desc'
    }

    let callCount = 0
    const resolvers: Array<(value: any) => void> = []

    apiModule.__setPaginatedDocumentsPostForTests((_request, controller) => {
      callCount += 1

      return new Promise((resolve, reject) => {
        resolvers.push(resolve)
        controller.signal.addEventListener(
          'abort',
          () => reject(new DOMException('Aborted', 'AbortError')),
          { once: true }
        )
      })
    })

    await expect(
      apiModule.getDocumentsPaginatedWithTimeout(request, 1)
    ).rejects.toThrow('Document fetch timeout')

    expect(callCount).toBe(1)

    const retryRequest = apiModule.getDocumentsPaginated(request)
    expect(callCount).toBe(2)

    resolvers[1]({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })

    await expect(retryRequest).resolves.toEqual({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })
  })

  test('does not abort a shared request when only one timeout subscriber expires', async () => {
    const request: DocumentsRequest = {
      status_filter: null,
      page: 1,
      page_size: 20,
      sort_field: 'updated_at',
      sort_direction: 'desc'
    }

    let callCount = 0
    let resolveSharedRequest: ((value: any) => void) | null = null
    let abortCount = 0

    apiModule.__setPaginatedDocumentsPostForTests((_request, controller) => {
      callCount += 1

      return new Promise((resolve, reject) => {
        resolveSharedRequest = resolve
        controller.signal.addEventListener(
          'abort',
          () => {
            abortCount += 1
            reject(new DOMException('Aborted', 'AbortError'))
          },
          { once: true }
        )
      })
    })

    const shortTimeoutRequest = apiModule.getDocumentsPaginatedWithTimeout(request, 1)
    const longTimeoutRequest = apiModule.getDocumentsPaginatedWithTimeout(request, 100)

    await expect(shortTimeoutRequest).rejects.toThrow('Document fetch timeout')

    expect(callCount).toBe(1)
    expect(abortCount).toBe(0)

    resolveSharedRequest?.({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })

    await expect(longTimeoutRequest).resolves.toEqual({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })
  })

  // The dedupe key is JSON.stringify(request), so file_path_filter must take
  // part in it: sharing an in-flight request across two different searches
  // would show one search's results under the other's term.
  test('treats requests differing only by file_path_filter as distinct', async () => {
    const base: DocumentsRequest = {
      status_filter: null,
      page: 1,
      page_size: 20,
      sort_field: 'updated_at',
      sort_direction: 'desc',
      file_path_filter: null
    }

    const seenFilters: Array<string | null | undefined> = []
    apiModule.__setPaginatedDocumentsPostForTests((request: any) => {
      seenFilters.push(request.file_path_filter)
      return new Promise(() => {})
    })

    apiModule.getDocumentsPaginated(base)
    apiModule.getDocumentsPaginated({ ...base, file_path_filter: 'invoice' })
    apiModule.getDocumentsPaginated({ ...base, file_path_filter: 'report' })

    expect(seenFilters).toEqual([null, 'invoice', 'report'])

    // Same filter again must still share the in-flight request
    apiModule.getDocumentsPaginated({ ...base, file_path_filter: 'report' })
    expect(seenFilters).toHaveLength(3)
  })
})
