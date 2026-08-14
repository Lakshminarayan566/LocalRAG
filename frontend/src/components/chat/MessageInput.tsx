import React, { useState, KeyboardEvent } from 'react';
import { Send, Sparkles } from 'lucide-react';
import { TaskType } from '../../types/api';

interface MessageInputProps {
  onSend: (message: string, taskType: TaskType) => void;
  disabled: boolean;
}

const TASK_OPTIONS: { value: TaskType; label: string }[] = [
  { value: 'general', label: 'General Q&A' },
  { value: 'explain', label: 'Explain Code' },
  { value: 'find_bugs', label: 'Find Bugs' },
  { value: 'similar_code', label: 'Similar Code' },
  { value: 'function_search', label: 'Search Functions' },
  { value: 'class_search', label: 'Search Classes' },
];

export const MessageInput: React.FC<MessageInputProps> = ({ onSend, disabled }) => {
  const [input, setInput] = useState('');
  const [taskType, setTaskType] = useState<TaskType>('general');

  const handleSend = () => {
    if (!input.trim() || disabled) return;
    onSend(input.trim(), taskType);
    setInput('');
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="max-w-4xl w-full mx-auto p-4">
      <div className="bg-surface border border-border rounded-xl p-3 shadow-xl focus-within:border-accent transition-colors">
        {/* Task Selector */}
        <div className="flex items-center space-x-2 mb-2 px-1">
          <Sparkles className="w-3.5 h-3.5 text-accent-light" />
          <span className="text-xs text-gray-400">Task Focus:</span>
          <select
            value={taskType}
            onChange={(e) => setTaskType(e.target.value as TaskType)}
            disabled={disabled}
            className="bg-background text-xs text-gray-200 border border-border rounded px-2 py-0.5 focus:outline-none focus:border-accent cursor-pointer"
          >
            {TASK_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {/* Text Input */}
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything about your codebase... (Shift + Enter for new line)"
          rows={2}
          disabled={disabled}
          className="w-full bg-transparent text-gray-100 placeholder-gray-500 text-sm focus:outline-none resize-none px-1"
        />

        {/* Actions */}
        <div className="flex items-center justify-between pt-2 border-t border-border/50">
          <span className="text-xs text-gray-500">PrivaRepo AI • Private & Local</span>
          <button
            onClick={handleSend}
            disabled={!input.trim() || disabled}
            className="bg-accent hover:bg-accent-hover disabled:bg-gray-800 disabled:text-gray-600 text-white p-2 rounded-lg transition-colors flex items-center justify-center"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};