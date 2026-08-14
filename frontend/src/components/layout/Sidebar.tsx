import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  MessageSquare,
  FolderGit2,
  Settings,
  ShieldCheck,
  Database,
  HardDrive,
  Circle,
} from 'lucide-react';
import { useApp } from '../../context/AppContext';

const NAV_ITEMS = [
  { to: '/chat', label: 'Chat', icon: MessageSquare },
  { to: '/repositories', label: 'Repositories', icon: FolderGit2 },
  { to: '/settings', label: 'Settings', icon: Settings },
];

export const Sidebar: React.FC = () => {
  const { health, activeRepo, repositories } = useApp();

  const isOllamaUp = health?.ollama_reachable ?? false;
  const activeRepoDetails = repositories.find((r) => r.name === activeRepo);

  return (
    <aside className="w-64 bg-surface/60 border-r border-white/[0.06] flex flex-col justify-between h-screen select-none">
      <div className="flex flex-col min-h-0">
        {/* App header */}
        <div className="px-4 py-4 border-b border-white/[0.06]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-accent/30 to-accent/5 border border-accent/30 flex items-center justify-center">
              <ShieldCheck className="w-5 h-5 text-accent-light" strokeWidth={2} />
            </div>
            <div className="min-w-0">
              <h1 className="font-semibold text-[15px] text-white tracking-tight leading-tight">
                PrivaRepo
              </h1>
              <p className="text-[11px] text-gray-500 leading-tight">
                Local Code Intelligence
              </p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="px-2.5 py-3 space-y-0.5">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `group relative flex items-center gap-3 px-3 py-2.5 rounded-lg text-[13px] font-medium transition-all ${
                  isActive
                    ? 'bg-accent/15 text-white'
                    : 'text-gray-400 hover:bg-white/[0.04] hover:text-gray-200'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <span
                    className={`absolute left-0 top-1/2 -translate-y-1/2 w-[3px] rounded-full transition-all ${
                      isActive ? 'h-4 bg-accent-light' : 'h-0 bg-transparent'
                    }`}
                  />
                  <Icon
                    className={`w-4 h-4 shrink-0 transition-colors ${
                      isActive ? 'text-accent-light' : 'text-gray-500 group-hover:text-gray-300'
                    }`}
                    strokeWidth={2}
                  />
                  <span>{label}</span>
                </>
              )}
            </NavLink>
          ))}
        </nav>
      </div>

      {/* Status footer */}
      <div className="px-3.5 py-4 border-t border-white/[0.06] space-y-2.5">
        {/* Active repository */}
        <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] p-3">
          <div className="flex items-center gap-1.5 text-[10.5px] text-gray-500 uppercase tracking-wide mb-1.5">
            <HardDrive className="w-3 h-3" />
            <span>Active Repository</span>
          </div>
          <p
            className="text-[13px] font-medium text-white truncate"
            title={activeRepoDetails?.name ?? undefined}
          >
            {activeRepoDetails ? activeRepoDetails.name : 'None selected'}
          </p>
        </div>

        {/* Ollama status */}
        <div className="flex items-center justify-between px-1 py-1">
          <div className="flex items-center gap-2 text-[12px] text-gray-400">
            <Database className="w-3.5 h-3.5" />
            <span>Ollama</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Circle
              className={`w-2 h-2 ${
                isOllamaUp
                  ? 'fill-emerald-500 text-emerald-500 animate-pulse'
                  : 'fill-red-500 text-red-500'
              }`}
            />
            <span
              className={`text-[11px] font-medium ${
                isOllamaUp ? 'text-emerald-400' : 'text-red-400'
              }`}
            >
              {isOllamaUp ? 'Connected' : 'Offline'}
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
};
