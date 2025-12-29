<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { marked } from 'marked'
import { listDocs, getDocContent, parseMarkdown, search, getPassword, setPassword, AuthError, ingest, deleteDoc, updateDoc, getDownloadUrl } from './api'

interface Doc {
  id: string
  path: string
  title: string | null
  createdAt?: string
  frontmatter?: Record<string, unknown>
  fileType?: 'text' | 'binary'
  size?: number
}

const docs = ref<Doc[]>([])
const filter = ref('')
const selectedDoc = ref<Doc | null>(null)
const docContent = ref('')
const docFrontmatter = ref<Record<string, unknown>>({})
const indexContent = ref('')
const loading = ref(false)
const searchQuery = ref('')
const searchResults = ref<Array<{ id: string; path: string; title: string | null; snippet: string }>>([])
const authenticated = ref(false)
const checkingAuth = ref(true)
const passwordInput = ref('')
const authError = ref('')
const showEditor = ref(false)
const editorTitle = ref('')
const editorContent = ref('')
const saving = ref(false)
const editingDoc = ref<Doc | null>(null)
const shareStatus = ref<'idle' | 'saving' | 'saved' | 'error'>('idle')
const sidebarOpen = ref(false)

const filteredDocs = computed(() => {
  if (!filter.value) return docs.value
  const q = filter.value.toLowerCase()
  return docs.value.filter(d =>
    d.title?.toLowerCase().includes(q) || d.path.toLowerCase().includes(q)
  )
})

async function login() {
  if (!passwordInput.value.trim()) return
  setPassword(passwordInput.value.trim())
  passwordInput.value = ''
  authError.value = ''
  await loadDocs()
}

async function loadDocs() {
  loading.value = true
  try {
    docs.value = await listDocs()
    authenticated.value = true
    // Try to load index.md
    try {
      const content = await getDocContent('index.md')
      const { body } = parseMarkdown(content)
      indexContent.value = body
    } catch {
      indexContent.value = '# Welcome to Whorl\n\nSelect a document from the sidebar.'
    }
  } catch (e) {
    if (e instanceof AuthError) {
      authenticated.value = false
      authError.value = 'Invalid password'
    } else {
      console.error('Failed to load docs:', e)
    }
  }
  loading.value = false
}

async function selectDoc(doc: Doc, pushUrl = true) {
  // Binary files: download directly
  if (doc.fileType === 'binary') {
    window.open(getDownloadUrl(doc.path), '_blank')
    return
  }

  selectedDoc.value = doc
  showEditor.value = false
  sidebarOpen.value = false
  if (pushUrl) updateUrl(doc.path)
  loading.value = true
  try {
    const content = await getDocContent(doc.path)
    const { frontmatter, body } = parseMarkdown(content)
    docFrontmatter.value = frontmatter
    docContent.value = body
  } catch (e) {
    docContent.value = 'Failed to load document.'
    docFrontmatter.value = {}
  }
  loading.value = false
}

function goHome(pushUrl = true) {
  selectedDoc.value = null
  docContent.value = ''
  showEditor.value = false
  if (pushUrl) updateUrl(null)
}

function openEditor() {
  showEditor.value = true
  selectedDoc.value = null
  editingDoc.value = null
  editorTitle.value = ''
  editorContent.value = ''
}

async function editDoc() {
  if (!selectedDoc.value) return
  editingDoc.value = selectedDoc.value
  editorTitle.value = selectedDoc.value.title || ''
  const content = await getDocContent(selectedDoc.value.path)
  const { body } = parseMarkdown(content)
  editorContent.value = body
  showEditor.value = true
  selectedDoc.value = null
}

async function saveDoc() {
  if (!editorContent.value.trim()) return
  saving.value = true
  try {
    if (editingDoc.value) {
      await updateDoc(editingDoc.value.path, editorContent.value, editorTitle.value || undefined)
    } else {
      await ingest(editorContent.value, editorTitle.value || undefined)
    }
    showEditor.value = false
    editingDoc.value = null
    editorTitle.value = ''
    editorContent.value = ''
    await loadDocs()
  } catch (e) {
    console.error('Failed to save:', e)
  }
  saving.value = false
}

async function doSearch() {
  if (!searchQuery.value.trim()) {
    searchResults.value = []
    return
  }
  loading.value = true
  try {
    searchResults.value = await search(searchQuery.value)
  } catch (e) {
    console.error('Search failed:', e)
  }
  loading.value = false
}

function renderMarkdown(content: string): string {
  // Convert [[file]] references to links
  const linked = content.replace(/\[\[([^\]]+)\]\]/g, (_, file) => {
    // Library files (non-.md) get direct download links
    if (!file.endsWith('.md')) {
      const url = getDownloadUrl(file)
      const name = file.split('/').pop() || file
      return `<a href="${url}" target="_blank" class="file-link">${name}</a>`
    }
    // Docs get internal navigation links
    return `<a href="#" class="doc-link" data-path="${file}">${file.replace('.md', '')}</a>`
  })
  return marked(linked) as string
}

function formatDate(dateStr?: string): string {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

  if (diffDays === 0) return 'today'
  if (diffDays === 1) return 'yesterday'
  if (diffDays < 7) return `${diffDays}d ago`
  if (diffDays < 30) return `${Math.floor(diffDays / 7)}w ago`
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

async function handleDelete() {
  if (!selectedDoc.value) return
  if (!confirm(`Delete "${selectedDoc.value.title || selectedDoc.value.path}"?`)) return

  try {
    await deleteDoc(selectedDoc.value.path)
    selectedDoc.value = null
    docContent.value = ''
    await loadDocs()
  } catch (e) {
    console.error('Failed to delete:', e)
  }
}

function getDocPathFromUrl(): string | null {
  const path = window.location.pathname
  if (path.startsWith('/d/')) {
    return decodeURIComponent(path.slice(3))
  }
  return null
}

function getAddParams(): { title: string; content: string; url: string } | null {
  if (window.location.pathname !== '/add') return null
  const params = new URLSearchParams(window.location.search)
  const content = params.get('content') || params.get('text') || ''
  const title = params.get('title') || ''
  const url = params.get('url') || ''
  if (!content && !title) return null
  return { title, content, url }
}

function getShareParams(): { title: string; text: string; url: string } | null {
  if (window.location.pathname !== '/share') return null
  const params = new URLSearchParams(window.location.search)
  const text = params.get('text') || ''
  const title = params.get('title') || ''
  const url = params.get('url') || ''
  if (!text && !url) return null
  return { title, text, url }
}

function updateUrl(docPath: string | null) {
  const url = docPath ? `/d/${encodeURIComponent(docPath)}` : '/'
  window.history.pushState({}, '', url)
}

async function loadDocFromUrl() {
  const docPath = getDocPathFromUrl()
  if (docPath && docs.value.length) {
    const doc = docs.value.find(d => d.path === docPath)
    if (doc) {
      await selectDoc(doc, false)
    }
  }
}

onMounted(async () => {
  // If we have a stored password, assume valid and load directly
  if (getPassword()) {
    authenticated.value = true
    checkingAuth.value = false

    // Check for /share (instant save from mobile share sheet)
    const shareParams = getShareParams()
    if (shareParams) {
      shareStatus.value = 'saving'
      const { title, text, url } = shareParams
      const now = new Date()
      const autoTitle = title || `Note ${now.toLocaleDateString()} ${now.toLocaleTimeString()}`
      const content = url ? (text ? `${text}\n\n${url}` : url) : text
      try {
        await ingest(content, autoTitle)
        shareStatus.value = 'saved'
        await loadDocs()
        setTimeout(() => {
          shareStatus.value = 'idle'
          window.history.replaceState({}, '', '/')
        }, 1500)
      } catch {
        shareStatus.value = 'error'
      }
    }
    // Check for /add bookmarklet params
    else {
      const addParams = getAddParams()
      if (addParams) {
        const { title, content, url } = addParams
        const body = url ? `${content}\n\n---\nSource: ${url}` : content
        editorTitle.value = title
        editorContent.value = body.trim()
        showEditor.value = true
        window.history.replaceState({}, '', '/')
        loadDocs() // fire and forget
      } else {
        // Need docs loaded before navigating to URL
        await loadDocs()
        await loadDocFromUrl()
      }
    }
  } else {
    checkingAuth.value = false
  }

  // Handle browser back/forward
  window.addEventListener('popstate', () => {
    const docPath = getDocPathFromUrl()
    if (docPath) {
      const doc = docs.value.find(d => d.path === docPath)
      if (doc) selectDoc(doc, false)
    } else {
      goHome(false)
    }
  })

  // Handle doc-link clicks
  document.addEventListener('click', (e) => {
    const target = e.target as HTMLElement
    if (target.classList.contains('doc-link')) {
      e.preventDefault()
      const path = target.dataset.path
      const doc = docs.value.find(d => d.path === path)
      if (doc) selectDoc(doc)
    }
  })
})
</script>

<template>
  <!-- Share overlay -->
  <div v-if="shareStatus !== 'idle'" class="share-overlay">
    <div class="share-box">
      <span v-if="shareStatus === 'saving'">Saving...</span>
      <span v-else-if="shareStatus === 'saved'">Saved!</span>
      <span v-else-if="shareStatus === 'error'">Failed to save</span>
    </div>
  </div>

  <div v-if="checkingAuth" class="login-screen">
    <div class="login-box">
      <h1>whorl</h1>
      <p class="checking">...</p>
    </div>
  </div>

  <div v-else-if="!authenticated" class="login-screen">
    <div class="login-box">
      <h1>whorl</h1>
      <p v-if="authError" class="error">{{ authError }}</p>
      <input
        v-model="passwordInput"
        type="password"
        placeholder="Password"
        @keyup.enter="login"
        class="login-input"
      />
      <button @click="login" class="login-btn">Enter</button>
    </div>
  </div>

  <div v-else class="app">
    <!-- Mobile header -->
    <header class="mobile-header">
      <button class="menu-btn" @click="sidebarOpen = !sidebarOpen">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
          <rect y="3" width="20" height="2"/>
          <rect y="9" width="20" height="2"/>
          <rect y="15" width="20" height="2"/>
        </svg>
      </button>
      <h1 @click="goHome()">whorl</h1>
      <button class="new-btn" @click="openEditor" title="New document">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="8" y1="3" x2="8" y2="13"></line>
          <line x1="3" y1="8" x2="13" y2="8"></line>
        </svg>
      </button>
    </header>

    <!-- Sidebar backdrop for mobile -->
    <div v-if="sidebarOpen" class="sidebar-backdrop" @click="sidebarOpen = false"></div>

    <aside class="sidebar" :class="{ open: sidebarOpen }">
      <div class="sidebar-header">
        <h1 @click="goHome(); sidebarOpen = false">whorl</h1>
        <button class="new-btn" @click="openEditor" title="New document">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="8" y1="3" x2="8" y2="13"></line>
            <line x1="3" y1="8" x2="13" y2="8"></line>
          </svg>
        </button>
      </div>

      <input
        v-model="filter"
        type="text"
        placeholder="Filter docs..."
        class="filter-input"
      />

      <nav class="doc-list">
        <a
          v-for="doc in filteredDocs"
          :key="doc.id"
          :href="doc.fileType === 'binary' ? getDownloadUrl(doc.path) : '/d/' + encodeURIComponent(doc.path)"
          :class="{ active: selectedDoc?.id === doc.id, binary: doc.fileType === 'binary' }"
          @click.prevent="selectDoc(doc)"
        >
          <span v-if="doc.fileType === 'binary'" class="file-indicator">↓</span>
          <span class="doc-title">{{ doc.title || doc.path }}</span>
          <span v-if="doc.createdAt" class="doc-date">{{ formatDate(doc.createdAt) }}</span>
        </a>
      </nav>

      <div class="search-section">
        <input
          v-model="searchQuery"
          type="text"
          placeholder="Search..."
          @keyup.enter="doSearch"
          class="search-input"
        />
        <div v-if="searchResults.length" class="search-results">
          <a
            v-for="r in searchResults"
            :key="r.id"
            :href="'/d/' + encodeURIComponent(r.path)"
            @click.prevent="selectDoc({ id: r.id, path: r.path, title: r.title })"
          >
            <strong>{{ r.title || r.path }}</strong>
            <span class="snippet">{{ r.snippet.slice(0, 80) }}...</span>
          </a>
        </div>
      </div>
    </aside>

    <main class="content">
      <div v-if="loading" class="loading">Loading...</div>

      <article v-else-if="showEditor" class="editor">
        <input
          v-model="editorTitle"
          type="text"
          placeholder="Title"
          class="editor-title"
        />
        <div class="editor-panes">
          <textarea
            v-model="editorContent"
            placeholder="Write markdown..."
            class="editor-content"
          ></textarea>
          <div class="editor-preview markdown" v-html="renderMarkdown(editorContent || '*preview*')"></div>
        </div>
        <div class="editor-actions">
          <button @click="showEditor = false; editingDoc = null" class="cancel-btn">Cancel</button>
          <button @click="saveDoc" :disabled="saving" class="save-btn">
            {{ saving ? 'Saving...' : editingDoc ? 'Update' : 'Save' }}
          </button>
        </div>
      </article>

      <article v-else-if="selectedDoc">
        <header class="doc-header">
          <h1 v-if="docFrontmatter.title">{{ docFrontmatter.title }}</h1>
          <div class="doc-meta">
            <span v-if="docFrontmatter.created_at" class="meta-item">
              {{ new Date(docFrontmatter.created_at as string).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }) }}
            </span>
            <span v-if="docFrontmatter.source" class="meta-item meta-source">{{ docFrontmatter.source }}</span>
          </div>
          <div v-if="docFrontmatter.tags" class="doc-tags">
            <span v-for="tag in (docFrontmatter.tags as string[])" :key="tag" class="tag">{{ tag }}</span>
          </div>
          <p v-if="docFrontmatter.summary" class="doc-summary">{{ docFrontmatter.summary }}</p>
        </header>
        <div class="markdown" v-html="renderMarkdown(docContent)"></div>
        <div class="doc-actions">
          <button @click="editDoc" class="edit-btn">Edit</button>
          <button @click="handleDelete" class="delete-btn">Delete</button>
        </div>
      </article>

      <article v-else>
        <div class="markdown" v-html="renderMarkdown(indexContent)"></div>
      </article>
    </main>
  </div>
</template>

<style>
@import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&family=Inter:wght@400;500;600&display=swap');

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html, body {
  font-family: "Courier Prime", monospace;
  font-size: 16px;
  line-height: 1.6;
  color: #333;
  background: #ded5bd;
}

.app {
  display: flex;
  min-height: 100vh;
  background: #ded5bd;
}

.sidebar {
  width: 260px;
  background: #d4cbb3;
  display: flex;
  flex-direction: column;
  position: fixed;
  top: 0;
  left: 0;
  height: 100vh;
  z-index: 100;
}

.sidebar-header {
  padding: 1.5rem;
  background: #2a2520;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.sidebar-header h1 {
  font-family: "Courier Prime", monospace;
  font-size: 1.5rem;
  font-weight: 700;
  cursor: pointer;
  color: #fff;
  text-transform: lowercase;
}

.new-btn {
  background: transparent;
  border: none;
  color: #fff;
  width: 24px;
  height: 24px;
  padding: 0;
  cursor: pointer;
  opacity: 0.7;
  transition: opacity 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.new-btn:hover {
  opacity: 1;
}

.filter-input,
.search-input {
  width: calc(100% - 1.5rem);
  margin: 0.75rem;
  padding: 0.5rem 0.75rem;
  border: 1px dotted #333;
  border-radius: 0;
  font-size: 0.9rem;
  font-family: "Courier Prime", monospace;
  background: #ded5bd;
  color: #333;
  outline: none;
}

.filter-input:focus,
.search-input:focus {
  border-style: solid;
}

.doc-list {
  flex: 1;
  overflow-y: auto;
  padding: 0.5rem 0;
  min-height: 0;
}

.doc-list a {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 0.5rem 1rem;
  color: #333;
  text-decoration: none;
  font-size: 0.9rem;
  cursor: pointer;
  transition: background 0.15s;
}

.doc-list a:hover {
  background: rgba(0,0,0,0.05);
}

.doc-list a.active {
  background: rgba(0,0,0,0.08);
}

.doc-list a.active .doc-title {
  font-weight: 700;
}

.doc-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.doc-date {
  font-size: 0.75rem;
  color: #888;
  margin-left: 0.5rem;
  flex-shrink: 0;
}

.file-indicator {
  font-size: 0.8rem;
  margin-right: 0.4rem;
  opacity: 0.6;
}

.doc-list a.binary {
  opacity: 0.8;
}

.search-section {
  border-top: 1px dotted #333;
  padding-bottom: 0.75rem;
}

.search-results {
  max-height: 200px;
  overflow-y: auto;
}

.search-results a {
  display: block;
  padding: 0.5rem 1rem;
  cursor: pointer;
  font-size: 0.85rem;
}

.search-results a:hover {
  background: rgba(0,0,0,0.05);
}

.search-results .snippet {
  display: block;
  color: #666;
  font-size: 0.75rem;
  margin-top: 0.25rem;
}

.content {
  flex: 1;
  margin-left: 260px;
  padding: 3rem 4rem;
  background: #ded5bd;
}

.content article {
  max-width: 700px;
}

.content .editor {
  max-width: none;
}

.loading {
  color: #666;
  font-style: italic;
}

article h1 {
  font-family: "Inter", -apple-system, sans-serif;
  font-size: 1.6rem;
  font-weight: 400;
  margin-bottom: 1.5rem;
  color: #333;
  letter-spacing: -0.01em;
}

.markdown h1 {
  font-family: "Inter", -apple-system, sans-serif;
  font-size: 1.6rem;
  font-weight: 400;
  margin: 2rem 0 1rem;
}
.markdown h2 {
  font-family: "Inter", -apple-system, sans-serif;
  font-size: 1.4rem;
  font-weight: 400;
  margin: 2.5rem 0 1rem;
}
.markdown h3 {
  font-family: "Inter", -apple-system, sans-serif;
  font-size: 1.2rem;
  font-weight: 400;
  margin: 2rem 0 0.75rem;
}
.markdown p {
  margin: 1rem 0 1.5rem;
  line-height: 1.7;
}
.markdown ul, .markdown ol {
  margin: 1rem 0 1.5rem;
  padding-left: 1.75rem;
}
.markdown li {
  margin: 0.5rem 0;
  line-height: 1.7;
}
.markdown code {
  font-family: "Courier Prime", monospace;
  background: rgba(0,0,0,0.08);
  padding: 0.15rem 0.4rem;
  border-radius: 3px;
  font-size: 0.9em;
  color: #333;
}
.markdown pre {
  background: #282828;
  color: #ebdbb2;
  padding: 1rem;
  border-radius: 5px;
  overflow-x: auto;
  margin: 1.5rem 0;
  line-height: 1.5;
}
.markdown pre code {
  background: none;
  padding: 0;
  color: #ebdbb2;
}
.markdown a, .doc-link, .file-link {
  color: #333;
  text-decoration: none;
  border-bottom: 1px dotted #333;
  transition: opacity 0.2s;
}
.markdown a:hover, .doc-link:hover, .file-link:hover {
  opacity: 0.7;
}
.file-link::before {
  content: "↓ ";
  font-size: 0.85em;
}
.markdown blockquote {
  border-left: 2px solid #333;
  padding-left: 1rem;
  margin: 1.5rem 0;
  color: #555;
  font-style: italic;
}
.markdown strong {
  font-weight: 700;
}
.markdown hr {
  border: none;
  border-top: 1px dotted #333;
  margin: 2rem 0;
}

.login-screen {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #ded5bd;
}

.login-box {
  background: #d4cbb3;
  padding: 3rem;
  text-align: center;
  min-width: 300px;
  border: 1px dotted #333;
}

.login-box h1 {
  font-family: "Courier Prime", monospace;
  font-size: 1.75rem;
  font-weight: 700;
  margin-bottom: 1.5rem;
  color: #333;
  text-transform: lowercase;
}

.login-box .error {
  color: #a33;
  font-size: 0.875rem;
  margin-bottom: 1rem;
}

.login-input {
  width: 100%;
  padding: 0.75rem;
  border: 1px dotted #333;
  border-radius: 0;
  font-size: 1rem;
  font-family: "Courier Prime", monospace;
  margin-bottom: 1rem;
  outline: none;
  background: #ded5bd;
}

.login-input:focus {
  border-style: solid;
}

.login-btn {
  width: 100%;
  padding: 0.75rem;
  background: #2a2520;
  color: #fff;
  border: none;
  font-size: 1rem;
  font-family: "Courier Prime", monospace;
  cursor: pointer;
  transition: background 0.2s;
}

.login-btn:hover {
  background: #3a352f;
}

.editor {
  display: flex;
  flex-direction: column;
  min-height: calc(100vh - 6rem);
}

.editor-title {
  font-family: "Inter", -apple-system, sans-serif;
  font-size: 1.6rem;
  font-weight: 400;
  border: none;
  background: transparent;
  padding: 0;
  width: 100%;
  outline: none;
  color: #333;
  letter-spacing: -0.01em;
  margin-bottom: 1rem;
}

.editor-title::placeholder {
  color: #999;
}

.editor-panes {
  display: flex;
  gap: 2rem;
  flex: 1;
  min-height: 400px;
}

.editor-content {
  flex: 1;
  font-family: "Courier Prime", monospace;
  font-size: 0.95rem;
  line-height: 1.7;
  border: none;
  background: rgba(255,255,255,0.3);
  padding: 1rem;
  border-radius: 4px;
  resize: none;
  outline: none;
  color: #333;
}

.editor-content:focus {
  background: rgba(255,255,255,0.5);
}

.editor-content::placeholder {
  color: #999;
}

.editor-preview {
  flex: 1;
  padding: 0 1rem;
  overflow-y: auto;
  opacity: 0.7;
  border-left: 1px dotted #999;
}

.editor-actions {
  display: flex;
  justify-content: flex-start;
  gap: 0.75rem;
  margin-top: 1.5rem;
  padding-top: 1rem;
}

.cancel-btn {
  padding: 0.5rem 1rem;
  background: transparent;
  border: 1px dotted #333;
  color: #333;
  font-family: "Courier Prime", monospace;
  font-size: 0.9rem;
  cursor: pointer;
}

.cancel-btn:hover {
  background: rgba(0,0,0,0.05);
}

.save-btn {
  padding: 0.5rem 1rem;
  background: #2a2520;
  border: none;
  color: #fff;
  font-family: "Courier Prime", monospace;
  font-size: 0.9rem;
  cursor: pointer;
}

.save-btn:hover {
  background: #3a352f;
}

.save-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.doc-header {
  margin-bottom: 2rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px dotted #999;
}

.doc-header h1 {
  font-family: "Inter", -apple-system, sans-serif;
  font-size: 1.8rem;
  font-weight: 500;
  margin: 0 0 0.75rem;
  color: #333;
}

.doc-meta {
  display: flex;
  gap: 1rem;
  font-size: 0.85rem;
  color: #666;
  margin-bottom: 0.75rem;
}

.meta-source {
  background: rgba(0,0,0,0.06);
  padding: 0.15rem 0.5rem;
  border-radius: 3px;
}

.doc-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}

.tag {
  font-size: 0.8rem;
  background: #2a2520;
  color: #fff;
  padding: 0.2rem 0.6rem;
  border-radius: 3px;
}

.doc-summary {
  font-size: 0.95rem;
  color: #555;
  font-style: italic;
  margin: 0;
  line-height: 1.6;
}

.doc-actions {
  margin-top: 3rem;
  padding-top: 1rem;
  border-top: 1px dotted #999;
  display: flex;
  gap: 0.75rem;
}

.edit-btn,
.delete-btn {
  padding: 0.4rem 0.75rem;
  background: transparent;
  border: 1px dotted #999;
  color: #888;
  font-family: "Courier Prime", monospace;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.2s;
}

.edit-btn:hover {
  border-color: #333;
  color: #333;
}

.delete-btn:hover {
  border-color: #a33;
  color: #a33;
}

.share-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.share-box {
  background: #2a2520;
  color: #fff;
  padding: 2rem 3rem;
  font-size: 1.25rem;
  font-family: "Courier Prime", monospace;
}

/* Mobile header - hidden on desktop */
.mobile-header {
  display: none;
}

.sidebar-backdrop {
  display: none;
}

/* Mobile styles */
@media (max-width: 768px) {
  .mobile-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem;
    background: #2a2520;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 200;
  }

  .mobile-header h1 {
    font-family: "Courier Prime", monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: #fff;
    text-transform: lowercase;
    cursor: pointer;
    margin: 0;
  }

  .menu-btn {
    background: transparent;
    border: none;
    color: #fff;
    padding: 0.25rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .sidebar-backdrop {
    display: block;
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    z-index: 250;
  }

  .sidebar {
    position: fixed;
    top: 0;
    left: 0;
    height: 100vh;
    width: 280px;
    transform: translateX(-100%);
    transition: transform 0.25s ease;
    z-index: 300;
  }

  .sidebar.open {
    transform: translateX(0);
  }

  .content {
    margin-left: 0;
    padding: 1.5rem;
    padding-top: calc(60px + 1.5rem);
  }

  .content article {
    max-width: 100%;
  }

  .editor-panes {
    flex-direction: column;
  }

  .editor-preview {
    border-left: none;
    border-top: 1px dotted #999;
    padding-top: 1rem;
    margin-top: 1rem;
  }

  .doc-header h1 {
    font-size: 1.4rem;
  }

  .doc-meta {
    flex-wrap: wrap;
  }
}

</style>
