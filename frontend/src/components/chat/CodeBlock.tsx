import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Check, Copy, FileCode2 } from 'lucide-react';

interface CodeBlockProps {
  language: string;
  value: string;
  /** Optional — lets callers (e.g. retrieved-context chunks) show a real
   *  filename in the header instead of just the language name. */
  filename?: string;
  /** Optional — line numbers start here instead of 1, so a retrieved
   *  chunk's numbers can match its real position in the source file. */
  startLine?: number;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({
  language,
  value,
  filename,
  startLine,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(value);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className="group relative my-3 rounded-xl border border-white/10 bg-[#0a0d12] shadow-lg shadow-black/20 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between gap-3 px-3.5 py-2 bg-surface/90 border-b border-white/10">
        <div className="flex items-center gap-2 min-w-0">
          <FileCode2 className="w-3.5 h-3.5 text-accent-light shrink-0" />
          {filename ? (
            <>
              <span className="font-mono text-[12px] text-gray-300 truncate">
                {filename}
              </span>
              <span className="hidden sm:inline text-[10px] font-mono uppercase tracking-wider text-gray-600 shrink-0">
                · {language || 'text'}
              </span>
            </>
          ) : (
            <span className="font-mono text-[10px] uppercase tracking-wider text-gray-500">
              {language || 'text'}
            </span>
          )}
        </div>

        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-medium text-gray-400 hover:text-white hover:bg-white/5 active:scale-95 transition-all shrink-0"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-emerald-400">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code surface */}
      <div className="overflow-x-auto [&::-webkit-scrollbar]:h-1.5 [&::-webkit-scrollbar-thumb]:bg-white/10 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-track]:bg-transparent">
        <SyntaxHighlighter
          language={language || 'text'}
          style={vscDarkPlus}
          showLineNumbers
          startingLineNumber={startLine ?? 1}
          wrapLongLines={false}
          customStyle={{
            margin: 0,
            padding: '0.85rem 1rem',
            fontSize: '0.8125rem',
            lineHeight: 1.65,
            background: 'transparent',
          }}
          codeTagProps={{
            style: { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' },
          }}
          lineNumberStyle={{
            minWidth: '2.75em',
            paddingRight: '1.25em',
            color: 'rgba(148,163,184,0.3)',
            userSelect: 'none',
          }}
        >
          {value}
        </SyntaxHighlighter>
      </div>
    </div>
  );
};
