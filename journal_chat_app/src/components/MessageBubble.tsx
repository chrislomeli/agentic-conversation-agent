import { Message } from '../types'

interface MessageBubbleProps {
  message: Message
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const { role, content } = message

  if (role === 'system') {
    return (
      <div className="flex items-center gap-3 my-4 text-xs text-slate-500">
        <div className="flex-1 h-px bg-slate-800" />
        <span className="shrink-0">{content}</span>
        <div className="flex-1 h-px bg-slate-800" />
      </div>
    )
  }

  const isUser = role === 'user'

  return (
    <div className="w-full mb-6">
      <p className={`text-xs font-medium mb-1.5 ${isUser ? 'text-indigo-400' : 'text-slate-400'}`}>
        {isUser ? 'You' : 'Journal Agent'}
      </p>
      <div
        className={`
          w-full px-4 py-3 rounded-lg text-sm leading-relaxed
          ${isUser
            ? 'bg-slate-800 border border-slate-700 text-slate-100'
            : 'bg-slate-800/40 border border-slate-700/40 text-slate-100'
          }
        `}
      >
        {content === '' ? (
          <span className="flex gap-1 items-center h-4">
            <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0ms]" />
            <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:150ms]" />
            <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:300ms]" />
          </span>
        ) : (
          <span className="whitespace-pre-wrap">{content}</span>
        )}
      </div>
    </div>
  )
}
