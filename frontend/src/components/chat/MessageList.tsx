import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { CodeBlock } from './CodeBlock';
import { RetrievedChunkOut } from '../../types/api';
import {
  FileText,
  ChevronDown,
  ChevronRight,
  User,
  Bot,
  Clock,
  Zap,
  Gauge,
  Folder,
} from 'lucide-react';

export interface Message {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  chunks?: RetrievedChunkOut[];
  timings?: {
    retrieval?: number;
    generation?: number;
    total?: number;
  };
  isStreaming?: boolean;
}

interface MessageListProps {
  messages: Message[];
}

// Retrieved chunks may carry scoring/rank fields that vary by backend
// version. Reading them through this intersection type keeps rendering
// fully optional ("where available") without altering RetrievedChunkOut
// itself or breaking on chunks that don't have them.
type ScoredChunk = RetrievedChunkOut &
  Partial<{
    rerank_score: number;
    vector_score: number;
    bm25_score: number;
    final_rank: number;
    rank: number;
    score: number;
    language: string;
  }>;

const EXTENSION_TO_LANGUAGE: Record<string, string> = {
  py: 'python',
  js: 'javascript',
  jsx: 'jsx',
  ts: 'typescript',
  tsx: 'tsx',
  java: 'java',
  go: 'go',
  rb: 'ruby',
  rs: 'rust',
  c: 'c',
  cpp: 'cpp',
  cc: 'cpp',
  h: 'c',
  hpp: 'cpp',
  cs: 'csharp',
  php: 'php',
  sh: 'bash',
  yml: 'yaml',
  yaml: 'yaml',
  json: 'json',
  md: 'markdown',
  sql: 'sql',
};

function inferLanguage(chunk: ScoredChunk): string {
  if (chunk.language) return chunk.language;
  const ext = chunk.file_path?.split('.').pop()?.toLowerCase();
  return (ext && EXTENSION_TO_LANGUAGE[ext]) || 'text';
}

function formatSeconds(ms?: number): string | null {
  if (typeof ms !== 'number') return null;
  return `${(ms / 1000).toFixed(2)}s`;
}

export const MessageList: React.FC<MessageListProps> = ({ messages }) => {
  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 space-y-8 max-w-4xl mx-auto w-full [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:bg-white/10 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-track]:bg-transparent">
      {messages.map((msg) => (
        <MessageItem key={msg.id} message={msg} />
      ))}
    </div>
  );
};

const MessageItem: React.FC<{ message: Message }> = ({ message }) => {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.sender === 'user';

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-xl bg-accent/15 border border-accent/30 flex items-center justify-center text-accent-light shrink-0 mt-0.5">
          <Bot className="w-4 h-4" strokeWidth={2} />
        </div>
      )}

      <div
        className={`flex flex-col gap-2.5 min-w-0 ${
          isUser ? 'items-end max-w-2xl' : 'items-start max-w-3xl flex-1'
        }`}
      >
        {/* Main message bubble */}
        <div
          className={`text-sm leading-relaxed ${
            isUser
              ? 'px-4 py-3 rounded-2xl rounded-tr-md bg-accent text-white shadow-md shadow-accent/10'
              : 'px-4 py-3.5 rounded-2xl rounded-tl-md bg-surface border border-white/[0.06] text-gray-200 shadow-sm w-full'
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap break-words">{message.text}</p>
          ) : (
            <div className="space-y-3">
              <ReactMarkdown
                components={{
                  p: ({ children }) => (
                    <p className="leading-relaxed text-gray-200 [&:not(:last-child)]:mb-3">
                      {children}
                    </p>
                  ),
                  ul: ({ children }) => (
                    <ul className="list-disc pl-5 space-y-1 mb-3 text-gray-200">{children}</ul>
                  ),
                  ol: ({ children }) => (
                    <ol className="list-decimal pl-5 space-y-1 mb-3 text-gray-200">{children}</ol>
                  ),
                  strong: ({ children }) => (
                    <strong className="font-semibold text-white">{children}</strong>
                  ),
                  a: ({ children, href }) => (
                    <a
                      href={href}
                      target="_blank"
                      rel="noreferrer"
                      className="text-accent-light underline underline-offset-2 hover:text-white transition-colors"
                    >
                      {children}
                    </a>
                  ),
                  h1: ({ children }) => (
                    <h1 className="text-base font-semibold text-white mt-1 mb-2">{children}</h1>
                  ),
                  h2: ({ children }) => (
                    <h2 className="text-[15px] font-semibold text-white mt-1 mb-2">{children}</h2>
                  ),
                  h3: ({ children }) => (
                    <h3 className="text-sm font-semibold text-white mt-1 mb-1.5">{children}</h3>
                  ),
                  code({ className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    const codeText = String(children).replace(/\n$/, '');
                    return match ? (
                      <CodeBlock language={match[1]} value={codeText} />
                    ) : (
                      <code
                        className="bg-background px-1.5 py-0.5 rounded text-[13px] font-mono text-accent-light"
                        {...props}
                      >
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {message.text}
              </ReactMarkdown>

              {message.isStreaming && (
                <span className="inline-flex items-center gap-1 h-4 align-middle">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent-light animate-pulse" />
                  <span
                    className="w-1.5 h-1.5 rounded-full bg-accent-light animate-pulse"
                    style={{ animationDelay: '150ms' }}
                  />
                  <span
                    className="w-1.5 h-1.5 rounded-full bg-accent-light animate-pulse"
                    style={{ animationDelay: '300ms' }}
                  />
                </span>
              )}
            </div>
          )}
        </div>

        {/* Retrieved context — collapsible, IDE-styled */}
        {!isUser && message.chunks && message.chunks.length > 0 && (
          <div className="w-full rounded-xl border border-white/[0.06] bg-surface/40 overflow-hidden">
            <button
              onClick={() => setShowSources(!showSources)}
              className="w-full flex items-center justify-between px-3.5 py-2.5 text-xs text-gray-400 hover:bg-white/[0.03] hover:text-gray-200 transition-colors"
            >
              <div className="flex items-center gap-2">
                <FileText className="w-3.5 h-3.5 text-accent-light" />
                <span className="font-medium">
                  Retrieved context
                </span>
                <span className="px-1.5 py-0.5 rounded-md bg-white/5 text-[10px] font-mono text-gray-400">
                  {message.chunks.length}
                </span>
              </div>
              {showSources ? (
                <ChevronDown className="w-3.5 h-3.5" />
              ) : (
                <ChevronRight className="w-3.5 h-3.5" />
              )}
            </button>

            {showSources && (
              <div className="px-3 pb-3 pt-1 space-y-2.5 border-t border-white/[0.06] max-h-[28rem] overflow-y-auto [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:bg-white/10 [&::-webkit-scrollbar-thumb]:rounded-full">
                {message.chunks.map((rawChunk, idx) => {
                  const chunk = rawChunk as ScoredChunk;
                  const rank = chunk.final_rank ?? chunk.rank;
                  const score = chunk.rerank_score ?? chunk.score;
                  const fileName = chunk.file_path?.split('/').pop() ?? chunk.file_path;

                  return (
                    <div key={chunk.chunk_id || idx} className="space-y-1.5">
                      <div className="flex items-center justify-between gap-2 px-0.5">
                        <div className="flex items-center gap-1.5 min-w-0 text-[11px] text-gray-400">
                          <Folder className="w-3 h-3 shrink-0 text-gray-500" />
                          <span className="font-mono truncate" title={chunk.file_path}>
                            {chunk.file_path}
                          </span>
                        </div>
                        <div className="flex items-center gap-1.5 shrink-0">
                          {typeof rank === 'number' && (
                            <span className="px-1.5 py-0.5 rounded-md bg-accent/10 border border-accent/20 text-[10px] font-mono text-accent-light">
                              #{rank}
                            </span>
                          )}
                          {typeof score === 'number' && (
                            <span className="px-1.5 py-0.5 rounded-md bg-white/5 text-[10px] font-mono text-gray-400">
                              {score.toFixed(3)}
                            </span>
                          )}
                          <span className="px-1.5 py-0.5 rounded-md bg-white/5 text-[10px] font-mono text-gray-500">
                            L{chunk.start_line}–{chunk.end_line}
                          </span>
                        </div>
                      </div>
                      <CodeBlock
                        language={inferLanguage(chunk)}
                        value={chunk.code_content}
                        filename={fileName}
                        startLine={chunk.start_line}
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Timing footer */}
        {!isUser && message.timings?.total !== undefined && (
          <div className="flex items-center gap-3 text-[10.5px] text-gray-500 px-1 font-mono">
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {formatSeconds(message.timings.total)}
            </span>
            {message.timings.retrieval !== undefined && (
              <span className="flex items-center gap-1 text-gray-600">
                <Gauge className="w-3 h-3" />
                retrieval {formatSeconds(message.timings.retrieval)}
              </span>
            )}
            {message.timings.generation !== undefined && (
              <span className="flex items-center gap-1 text-gray-600">
                <Zap className="w-3 h-3" />
                generation {formatSeconds(message.timings.generation)}
              </span>
            )}
          </div>
        )}
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center text-gray-300 shrink-0 mt-0.5">
          <User className="w-4 h-4" strokeWidth={2} />
        </div>
      )}
    </div>
  );
};
