import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { HealthResponse, RepositoryResponse } from '../types/api';
import { getHealth, listRepositories, selectRepository } from '../services/api';

interface AppContextType {
  health: HealthResponse | null;
  repositories: RepositoryResponse[];
  activeRepo: string | null;
  loading: boolean;
  refreshState: () => Promise<void>;
  setActiveRepoName: (name: string) => Promise<void>;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [repositories, setRepositories] = useState<RepositoryResponse[]>([]);
  const [activeRepo, setActiveRepo] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const refreshState = useCallback(async () => {
    try {
      setLoading(true);
      const [healthData, repoListData] = await Promise.all([
        getHealth().catch(() => null),
        listRepositories().catch(() => ({ active: null, repositories: [] })),
      ]);

      if (healthData) setHealth(healthData);
      setRepositories(repoListData.repositories);
      setActiveRepo(repoListData.active);
    } catch (err) {
      console.error('Failed to load application state:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const setActiveRepoName = async (name: string) => {
    try {
      await selectRepository(name);
      await refreshState();
    } catch (err) {
      console.error('Failed to set active repository:', err);
      throw err;
    }
  };

  useEffect(() => {
    refreshState();
  }, [refreshState]);

  return (
    <AppContext.Provider
      value={{
        health,
        repositories,
        activeRepo,
        loading,
        refreshState,
        setActiveRepoName,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};