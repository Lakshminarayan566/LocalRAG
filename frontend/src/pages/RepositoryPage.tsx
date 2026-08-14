import React, { useState, useEffect } from 'react';
import { useApp } from '../context/AppContext';
import {
  addRepository,
  removeRepository,
  startIndexing,
  getIndexStatus,
  getStats,
} from '../services/api';
import { IndexJobStatus, StatsResponse } from '../types/api';
import {
  FolderGit2,
  Plus,
  Trash2,
  CheckCircle2,
  RefreshCw,
  Layers,
  Database,
  AlertCircle,
} from 'lucide-react';

export const RepositoryPage: React.FC = () => {
  const { repositories, activeRepo, refreshState, setActiveRepoName } = useApp();

  const [pathInput, setPathInput] = useState('');
  const [nameInput, setNameInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [currentIndexJob, setCurrentIndexJob] = useState<IndexJobStatus | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);

  // Load repository stats
  useEffect(() => {
    getStats()
      .then(setStats)
      .catch(() => setStats(null));
  }, [activeRepo]);

  // Handle Add Repository
  const handleAddRepo = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!pathInput.trim()) return;

    try {
      setLoading(true);
      setError(null);
      await addRepository({ path: pathInput.trim(), name: nameInput.trim() || undefined });
      setPathInput('');
      setNameInput('');
      await refreshState();
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to register repository');
    } finally {
      setLoading(false);
    }
  };

  // Handle Delete Repository
  const handleRemoveRepo = async (name: string) => {
    try {
      setLoading(true);
      await removeRepository(name, false);
      await refreshState();
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to remove repository');
    } finally {
      setLoading(false);
    }
  };

  // Trigger Indexing Job
  const handleIndex = async () => {
    if (!activeRepo) return;
    try {
      setError(null);
      const job = await startIndexing({ repo: activeRepo });
      setCurrentIndexJob(job);
      pollJobStatus(job.job_id);
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to start indexing job');
    }
  };

  // Poll Indexing Job Status
  const pollJobStatus = (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const job = await getIndexStatus(jobId);
        setCurrentIndexJob(job);
        if (job.status === 'completed' || job.status === 'failed') {
          clearInterval(interval);
          getStats().then(setStats);
          refreshState();
        }
      } catch {
        clearInterval(interval);
      }
    }, 1500);
  };

  return (
    <div className="flex-1 overflow-y-auto p-8 max-w-5xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white mb-1">Repository Management</h1>
        <p className="text-xs text-gray-400">
          Register local directory paths, switch active context, and trigger hybrid indexing.
        </p>
      </div>

      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl flex items-center space-x-2 text-red-400 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Add Repository Form */}
      <form onSubmit={handleAddRepo} className="bg-surface border border-border p-6 rounded-2xl space-y-4">
        <h2 className="text-sm font-semibold text-white flex items-center space-x-2">
          <Plus className="w-4 h-4 text-accent-light" />
          <span>Register New Local Repository</span>
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Absolute Path</label>
            <input
              type="text"
              placeholder="/home/user/projects/my-repo"
              value={pathInput}
              onChange={(e) => setPathInput(e.target.value)}
              className="w-full bg-background border border-border rounded-lg px-3 py-2 text-xs text-white placeholder-gray-600 focus:outline-none focus:border-accent"
              required
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Display Name (Optional)</label>
            <input
              type="text"
              placeholder="my-repo"
              value={nameInput}
              onChange={(e) => setNameInput(e.target.value)}
              className="w-full bg-background border border-border rounded-lg px-3 py-2 text-xs text-white placeholder-gray-600 focus:outline-none focus:border-accent"
            />
          </div>
        </div>
        <button
          type="submit"
          disabled={loading}
          className="bg-accent hover:bg-accent-hover text-white text-xs font-medium px-4 py-2 rounded-lg transition-colors flex items-center space-x-2"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>Add Repository</span>
        </button>
      </form>

      {/* Registered Repositories List */}
      <div className="space-y-4">
        <h2 className="text-sm font-semibold text-white">Registered Repositories</h2>
        <div className="grid grid-cols-1 gap-3">
          {repositories.map((repo) => {
            const isActive = repo.name === activeRepo;
            return (
              <div
                key={repo.name}
                className={`bg-surface border p-4 rounded-xl flex items-center justify-between transition-colors ${
                  isActive ? 'border-accent bg-accent/5' : 'border-border'
                }`}
              >
                <div className="space-y-1">
                  <div className="flex items-center space-x-2">
                    <FolderGit2 className="w-4 h-4 text-accent-light" />
                    <span className="font-semibold text-sm text-white">{repo.name}</span>
                    {isActive && (
                      <span className="bg-accent/20 border border-accent/40 text-accent-light text-[10px] px-2 py-0.5 rounded-full font-medium">
                        Active
                      </span>
                    )}
                  </div>
                  <p className="text-xs font-mono text-gray-400">{repo.path}</p>
                </div>

                <div className="flex items-center space-x-2">
                  {!isActive && (
                    <button
                      onClick={() => setActiveRepoName(repo.name)}
                      className="bg-surface-hover hover:bg-gray-700 text-gray-200 text-xs px-3 py-1.5 rounded-lg transition-colors"
                    >
                      Set Active
                    </button>
                  )}
                  <button
                    onClick={() => handleRemoveRepo(repo.name)}
                    className="p-1.5 text-gray-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Indexing Action & Stats */}
      {activeRepo && (
        <div className="bg-surface border border-border p-6 rounded-2xl space-y-6">
          <div className="flex items-center justify-between border-b border-border pb-4">
            <div>
              <h2 className="text-sm font-semibold text-white">Index Active Repository</h2>
              <p className="text-xs text-gray-400">
                Parse code structure with Tree-Sitter and generate vector/BM25 search indices.
              </p>
            </div>
            <button
              onClick={handleIndex}
              disabled={currentIndexJob?.status === 'running'}
              className="bg-accent hover:bg-accent-hover disabled:bg-gray-800 text-white text-xs font-medium px-4 py-2 rounded-lg transition-colors flex items-center space-x-2"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${currentIndexJob?.status === 'running' ? 'animate-spin' : ''}`} />
              <span>{currentIndexJob?.status === 'running' ? 'Indexing...' : 'Index Now'}</span>
            </button>
          </div>

          {/* Job Progress */}
          {currentIndexJob && (
            <div className="bg-background/60 p-3 rounded-lg border border-border text-xs space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Job ID: {currentIndexJob.job_id}</span>
                <span
                  className={`font-semibold ${
                    currentIndexJob.status === 'completed'
                      ? 'text-emerald-400'
                      : currentIndexJob.status === 'failed'
                      ? 'text-red-400'
                      : 'text-accent-light'
                  }`}
                >
                  {currentIndexJob.status.toUpperCase()}
                </span>
              </div>
            </div>
          )}

          {/* Index Stats */}
          {stats && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-background p-3 rounded-xl border border-border">
                <span className="text-[10px] text-gray-400 block">Total Chunks</span>
                <span className="text-base font-bold text-white">{stats.total_chunks}</span>
              </div>
              <div className="bg-background p-3 rounded-xl border border-border">
                <span className="text-[10px] text-gray-400 block">Unique Files</span>
                <span className="text-base font-bold text-white">{stats.unique_files ?? 0}</span>
              </div>
              <div className="bg-background p-3 rounded-xl border border-border">
                <span className="text-[10px] text-gray-400 block">BM25 Index Built</span>
                <span className="text-base font-bold text-white">
                  {stats.bm25_index_built ? 'Yes' : 'No'}
                </span>
              </div>
              <div className="bg-background p-3 rounded-xl border border-border">
                <span className="text-[10px] text-gray-400 block">Embedding Model</span>
                <span className="text-xs font-mono font-bold text-accent-light truncate block">
                  {stats.embedding_model ?? 'N/A'}
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};