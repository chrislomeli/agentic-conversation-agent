import { useState, useCallback, useRef } from 'react'
import { Message } from '../types'

const BASE_URL = '/api'  // proxied to http://localhost:8000 via vite.config.ts

function generateId(): string {
  return Math.random().toString(36).slice(2, 11)
}

export function useChat() {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const stop = useCallback(() => {
    abortControllerRef.current?.abort()
    setIsLoading(false)
  }, [])

  // Create a new server-side session; reuse existing one if already allocated.
  const ensureSession = useCallback(async (): Promise<string> => {
    if (sessionId) return sessionId
    const res = await fetch(`${BASE_URL}/sessions`, { method: 'POST' })
    if (!res.ok) throw new Error(`Failed to create session: ${res.status}`)
    const data = await res.json()
    setSessionId(data.session_id)
    return data.session_id as string
  }, [sessionId])

  // Run the end-of-session pipeline on the server then drop the local session.
  const endSession = useCallback(async (sid: string) => {
    try {
      await fetch(`${BASE_URL}/sessions/${sid}`, { method: 'DELETE' })
    } catch {
      // best-effort; don't block the UI
    }
    setSessionId(null)
  }, [])

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isLoading) return

    setError(null)

    const isQuit = content.trim().toLowerCase() === '/quit'

    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    // Reserve a placeholder for the streaming assistant reply (not for /quit).
    const assistantId = generateId()
    if (!isQuit) {
      setMessages(prev => [
        ...prev,
        { id: assistantId, role: 'assistant', content: '', timestamp: new Date() },
      ])
    }

    abortControllerRef.current = new AbortController()

    // Ensure a session exists before opening the stream.
    let sid: string | null = null
    try {
      sid = await ensureSession()
    } catch (err) {
      setError((err as Error).message ?? 'Could not create session')
      setIsLoading(false)
      return
    }

    try {
      const response = await fetch(`${BASE_URL}/chat/${sid}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortControllerRef.current.signal,
        body: JSON.stringify({ message: content.trim() }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      // ---------- Typed SSE streaming ----------
      // Wire format from the API:
      //   event: token\ndata: {"text": "chunk"}\n\n
      //   event: system\ndata: {"text": "..."}\n\n
      //   event: done\ndata: {"text": ""}\n\n
      //   event: error\ndata: {"text": "message"}\n\n
      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let currentEvent = 'token'

      outer: while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim()
          } else if (line.startsWith('data: ')) {
            let text = ''
            try {
              const parsed = JSON.parse(line.slice(6).trim())
              text = parsed.text ?? ''
            } catch {
              text = line.slice(6).trim()
            }

            if (currentEvent === 'token' && text) {
              setMessages(prev =>
                prev.map(m =>
                  m.id === assistantId ? { ...m, content: m.content + text } : m
                )
              )
            } else if (currentEvent === 'system' && text) {
              setMessages(prev => [
                ...prev,
                { id: generateId(), role: 'system', content: text, timestamp: new Date() },
              ])
            } else if (currentEvent === 'done') {
              break outer
            } else if (currentEvent === 'error') {
              setError(text || 'Server error')
              setMessages(prev => prev.filter(m => m.id !== assistantId))
              break outer
            }
          } else if (line === '') {
            // Blank line = SSE event boundary; reset event type to default.
            currentEvent = 'token'
          }
        }
      }
    } catch (err) {
      if ((err as Error).name === 'AbortError') return
      setError((err as Error).message ?? 'Something went wrong')
      setMessages(prev => prev.filter(m => m.id !== assistantId))
    } finally {
      setIsLoading(false)
    }

    // After a /quit the server has already run its cleanup stream; now trigger
    // the end-of-session pipeline and drop the local session reference.
    if (isQuit && sid) {
      await endSession(sid)
    }
  }, [isLoading, ensureSession, endSession])

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => setInput(e.target.value),
    []
  )

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault()
      sendMessage(input)
    },
    [input, sendMessage]
  )

  // Clear local messages and drop the session so the next send starts fresh.
  const clearMessages = useCallback(() => {
    setMessages([])
    setSessionId(null)
  }, [])

  return {
    messages,
    input,
    isLoading,
    error,
    handleInputChange,
    handleSubmit,
    sendMessage,
    stop,
    clearMessages,
    setInput,
  }
}
