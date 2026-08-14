import React from 'react';
import { Link } from 'react-router-dom';
import { MessageSquare, FolderGit2, ShieldCheck, Cpu, Zap, Search } from 'lucide-react';
import { useApp } from '../context/AppContext';

export const LandingPage: React.FC = () => {
  const { activeRepo, health } = useApp();

  return (
    <div className="flex-1 overflow-y-auto p-8 flex flex-col items-center justify-center max-w-5xl mx-auto">
      {/* Hero */}
      <div className="text-center space-y-4 mb-12">
        <div className="inline-flex items-center space-x-2 bg-accent/10 border border-accent/30 px-3 py-1 rounded-full text-xs text-accent-light font-medium">
          <ShieldCheck className="w-4 h-4" />
          <span>100% On-Premise & Local Execution</span>
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold text-white tracking-tight">
          Syntax-Aware Code Intelligence with <span className="text-accent-light">PrivaRepo</span>
        </h1>
        <p className="text-gray-400 max-w-2xl mx-auto text-base">
          Ask questions, discover implementations, and analyze your repositories using local hybrid retrieval (BM25 + Vector) and Ollama LLM integration.
        </p>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center space-x-4 mb-16">
        <Link
          to="/chat"
          className="bg-accent hover:bg-accent-hover text-white font-medium px-6 py-3 rounded-xl transition-all shadow-lg shadow-accent/20 flex items-center space-x-2"
        >
          <MessageSquare className="w-5 h-5" />
          <span>Start Chatting</span>
        </Link>

        <Link
          to="/repositories"
          className="bg-surface hover:bg-surface-hover border border-border text-gray-200 font-medium px-6 py-3 rounded-xl transition-all flex items-center space-x-2"
        >
          <FolderGit2 className="w-5 h-5" />
          <span>Manage Repositories</span>
        </Link>
      </div>

      {/* Feature Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full">
        <div className="bg-surface border border-border p-6 rounded-2xl space-y-3">
          <div className="bg-accent/20 p-3 rounded-xl w-fit text-accent-light border border-accent/30">
            <Search className="w-6 h-6" />
          </div>
          <h3 className="font-semibold text-lg text-white">Hybrid Retrieval</h3>
          <p className="text-xs text-gray-400 leading-relaxed">
            Combines Vector embeddings with BM25 keyword matching via Reciprocal Rank Fusion (RRF) for precise code search.
          </p>
        </div>

        <div className="bg-surface border border-border p-6 rounded-2xl space-y-3">
          <div className="bg-accent/20 p-3 rounded-xl w-fit text-accent-light border border-accent/30">
            <Cpu className="w-6 h-6" />
          </div>
          <h3 className="font-semibold text-lg text-white">Tree-Sitter Chunking</h3>
          <p className="text-xs text-gray-400 leading-relaxed">
            Parses AST structure to retain complete functions, classes, and method contexts instead of naive character splitting.
          </p>
        </div>

        <div className="bg-surface border border-border p-6 rounded-2xl space-y-3">
          <div className="bg-accent/20 p-3 rounded-xl w-fit text-accent-light border border-accent/30">
            <Zap className="w-6 h-6" />
          </div>
          <h3 className="font-semibold text-lg text-white">SSE Streaming</h3>
          <p className="text-xs text-gray-400 leading-relaxed">
            Real-time token generation streamed directly from your local Ollama server with step-by-step context transparency.
          </p>
        </div>
      </div>
    </div>
  );
};