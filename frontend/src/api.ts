const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const STORAGE_KEY = 'whorl_api_key'

export function getApiKey(): string {
  return localStorage.getItem(STORAGE_KEY) || ''
}

export function setApiKey(key: string) {
  localStorage.setItem(STORAGE_KEY, key)
}

export function clearApiKey() {
  localStorage.removeItem(STORAGE_KEY)
}

interface Doc {
  id: string
  path: string
  title: string | null
  createdAt?: string
  content?: string
  frontmatter?: Record<string, unknown>
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
      'X-API-Key': getApiKey(),
      ...options.headers,
    },
  })
  if (res.status === 401) {
    clearApiKey()
    throw new AuthError()
  }
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

export async function listDocs(): Promise<Doc[]> {
  const { stdout } = await request('/bash', {
    method: 'POST',
    body: JSON.stringify({ command: 'ls -1 *.md 2>/dev/null || true' }),
  })

  const files = stdout.trim().split('\n').filter(Boolean)
  const docs: Doc[] = []

  for (const file of files) {
    const content = await getDocContent(file)
    const { frontmatter } = parseMarkdown(content)
    docs.push({
      id: frontmatter.id as string || file.replace('.md', ''),
      path: file,
      title: frontmatter.title as string || file.replace('.md', ''),
      createdAt: frontmatter.created_at as string || undefined,
      frontmatter,
    })
  }

  // Sort by date, newest first
  docs.sort((a, b) => {
    if (!a.createdAt) return 1
    if (!b.createdAt) return -1
    return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  })

  return docs
}

export async function getDocContent(path: string): Promise<string> {
  const { stdout } = await request('/bash', {
    method: 'POST',
    body: JSON.stringify({ command: `cat "${path}"` }),
  })
  return stdout
}

export async function search(query: string, limit = 10): Promise<SearchResult[]> {
  const { results } = await request('/search', {
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

export function parseMarkdown(content: string): { frontmatter: Record<string, unknown>; body: string } {
  if (content.startsWith('---')) {
    const parts = content.split('---')
    if (parts.length >= 3) {
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
