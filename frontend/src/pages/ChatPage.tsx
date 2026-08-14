import React, { useState } from 'react';
import { useApp } from '../context/AppContext';
import { MessageList, Message } from '../components/chat/MessageList';
import { MessageInput } from '../components/chat/MessageInput';
import { streamChat } from '../services/api';
import { TaskType } from '../types/api';
import { AlertCircle, HardDrive, Sparkles, X, CircleDot } from 'lucide-react';

export const ChatPage: React.FC = () => {
  const { activeRepo, health } = useApp();
  const [messages, setMessages] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSend = async (question: string, taskType: TaskType) => {
    if (!activeRepo) {
      setError('Please select an active repository before initiating chat.');
      return;
    }

    setError(null);
    const userMsgId = Date.now().toString();
    const assistantMsgId = (Date.now() + 1).toString();

    const userMessage: Message = {
      id: userMsgId,
      sender: 'user',
      text: question,
    };

    const initialAssistantMessage: Message = {
      id: assistantMsgId,
      sender: 'assistant',
      text: '',
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMessage, initialAssistantMessage]);
    setIsGenerating(true);

    let currentText = '';

    await streamChat(
      { query: question, task_focus: taskType, repo_name: activeRepo },
      {
        onChunks: ({ results }) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId ? { ...msg, chunks: results } : msg
            )
          );
        },
        onToken: (token) => {
          currentText += token;
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId ? { ...msg, text: currentText } : msg
            )
          );
        },
        onDone: (data) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId
                ? {
                    ...msg,
                    isStreaming: false,
                    timings: {
                      retrieval: data.retrieval_time,
                      generation: data.generation_time,
                      total: data.total_time,
                    },
                  }
                : msg
            )
          );
          setIsGenerating(false);
        },
        onError: (errMessage) => {
          setError(errMessage);
          setIsGenerating(false);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId
                ? {
                    ...msg,
                    text:
                      currentText ||
                      'An error occurred while streaming response.',
                    isStreaming: false,
                  }
                : msg
            )
          );
        },
      }
    );
  };

  const isOllamaUp = health?.ollama_reachable ?? false;

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-background">
      {/* Top bar */}
      <div className="shrink-0 bg-surface/50 backdrop-blur-sm border-b border-white/[0.06] px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2.5 text-[12.5px]">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white/[0.04] border border-white/[0.06]">
            <HardDrive className="w-3.5 h-3.5 text-accent-light" />
            <span className="text-gray-500">Repository</span>
            <span className="font-semibold text-white">
              {activeRepo || 'None selected'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {isGenerating && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-accent/10 border border-accent/20 text-[11px] text-accent-light">
              <CircleDot className="w-3 h-3 animate-pulse" />
              <span>Generating</span>
            </div>
          )}
          <div className="flex items-center gap-1.5 text-[11px] text-gray-500">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                isOllamaUp ? 'bg-emerald-500' : 'bg-red-500'
              }`}
            />
            <span>{isOllamaUp ? 'Ollama connected' : 'Ollama offline'}</span>
          </div>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="mx-4 mt-4 shrink-0 flex items-start gap-2.5 px-3.5 py-2.5 rounded-xl bg-red-500/[0.08] border border-red-500/20 text-red-400 text-[12.5px]">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <span className="flex-1">{error}</span>
          <button
            onClick={() => setError(null)}
            className="shrink-0 p-0.5 rounded-md hover:bg-red-500/10 transition-colors"
            aria-label="Dismiss error"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Message list / empty state */}
      {messages.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center px-8 text-center">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-accent/25 to-accent/5 border border-accent/25 flex items-center justify-center text-accent-light mb-5">
            <Sparkles className="w-7 h-7" strokeWidth={1.75} />
          </div>
          <h2 className="text-xl font-semibold text-white mb-2 tracking-tight">
            Ready to explore your code
          </h2>
          <p className="text-[13px] text-gray-500 max-w-md leading-relaxed">
            Ask about classes, functions, architectural flow, or specific
            implementations in{' '}
            <span className="text-gray-300 font-medium">
              {activeRepo || 'your codebase'}
            </span>
            .
          </p>
        </div>
      ) : (
        <MessageList messages={messages} />
      )}

      {/* Input */}
      <div className="shrink-0">
        <MessageInput onSend={handleSend} disabled={isGenerating} />
      </div>
    </div>
  );
};
