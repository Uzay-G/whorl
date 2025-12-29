const API_BASE = import.meta.env.VITE_API_URL || '/api'
const STORAGE_KEY = 'whorl_password'

export function getPassword(): string {
  return localStorage.getItem(STORAGE_KEY) || ''
}

export function setPassword(password: string) {
  localStorage.setItem(STORAGE_KEY, password)
}

export function clearPassword() {
  localStorage.removeItem(STORAGE_KEY)
}

interface Doc {
  id: string
  path: string
  title: string | null
  createdAt?: string
  content?: string
  frontmatter?: Record<string, unknown>
  fileType?: 'text' | 'binary'
  size?: number
}

interface SearchResult {
  id: string
  path: string
  title: string | null
  snippet: string
  score: number
}

export class AuthError extends Error {
  constructor() {
    super('Unauthorized')
    this.name = 'AuthError'
  }
}

async function request(endpoint: string, options: RequestInit = {}) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'X-Password': getPassword(),
      ...options.headers,
    },
  })
  if (res.status === 401) {
    clearPassword()
    throw new AuthError()
  }
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

export async function listDocs(): Promise<Doc[]> {
  const { docs } = await request('/documents')
  return docs.map((d: Record<string, unknown>) => ({
    id: d.id as string,
    path: d.path as string,
    title: d.title as string || d.path as string,
    createdAt: d.created_at as string || undefined,
    fileType: d.file_type as 'text' | 'binary',
    size: d.size as number,
  }))
}

export async function getDocContent(path: string): Promise<string> {
  const { content } = await request(`/documents/${encodeURIComponent(path)}`)
  return content
}

export async function search(query: string, limit = 10): Promise<SearchResult[]> {
  const { results } = await request('/text_search', {
    method: 'POST',
    body: JSON.stringify({ query, limit }),
  })
  return results
}

export async function agentSearch(query: string): Promise<string> {
  const { answer } = await request('/agent_search', {
    method: 'POST',
    body: JSON.stringify({ query }),
  })
  return answer
}

export async function ingest(content: string, title?: string, process = false): Promise<{ id: string; path: string }> {
  return await request('/ingest', {
    method: 'POST',
    body: JSON.stringify({ content, title: title || null, process }),
  })
}

export async function deleteDoc(path: string): Promise<void> {
  await request('/delete', {
    method: 'POST',
    body: JSON.stringify({ path }),
  })
}

export async function updateDoc(path: string, content: string, title?: string): Promise<void> {
  await request('/update', {
    method: 'POST',
    body: JSON.stringify({ path, content, title }),
  })
}

export function getDownloadUrl(path: string): string {
  // Encode each path segment separately to preserve slashes
  const encodedPath = path.split('/').map(encodeURIComponent).join('/')
  return `${API_BASE}/download/${encodedPath}?password=${encodeURIComponent(getPassword())}`
}

export function parseMarkdown(content: string): { frontmatter: Record<string, unknown>; body: string } {
  if (content.startsWith('---')) {
    const parts = content.split('---')
    if (parts.length >= 3 && parts[1]) {
      const yamlStr = parts[1].trim()
      const body = parts.slice(2).join('---').trim()
      const frontmatter: Record<string, unknown> = {}

      for (const line of yamlStr.split('\n')) {
        const [key, ...valueParts] = line.split(':')
        if (key && valueParts.length) {
          let value: unknown = valueParts.join(':').trim()
          // Strip quotes from YAML values
          if (typeof value === 'string' && ((value.startsWith("'") && value.endsWith("'")) || (value.startsWith('"') && value.endsWith('"')))) {
            value = value.slice(1, -1)
          }
          if (typeof value === 'string' && value.startsWith('[') && value.endsWith(']')) {
            value = value.slice(1, -1).split(',').map(s => s.trim())
          }
          frontmatter[key.trim()] = value
        }
      }

      return { frontmatter, body }
    }
  }
  return { frontmatter: {}, body: content }
}
