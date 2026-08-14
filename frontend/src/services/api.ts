import axios from 'axios';
import {
  HealthResponse,
  RepositoryListResponse,
  RepositoryResponse,
  RepositoryAddRequest,
  IndexJobStatus,
  IndexRequest,
  StatsResponse,
  SearchRequest,
  SearchResponse,
  ChatRequest,
  ChatResponse,
  SettingsResponse,
  SettingsUpdateRequest,
  RetrievedChunkOut,
} from '../types/api';

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health
export const getHealth = async (): Promise<HealthResponse> => {
  const res = await api.get<HealthResponse>('/health');
  return res.data;
};

// Repositories
export const listRepositories = async (): Promise<RepositoryListResponse> => {
  const res = await api.get<RepositoryListResponse>('/repositories');
  return res.data;
};

export const addRepository = async (
  payload: RepositoryAddRequest
): Promise<RepositoryResponse> => {
  const res = await api.post<RepositoryResponse>('/repositories', payload);
  return res.data;
};

export const selectRepository = async (
  name: string
): Promise<RepositoryResponse> => {
  const res = await api.post<RepositoryResponse>(`/repositories/${name}/select`);
  return res.data;
};

export const removeRepository = async (
  name: string,
  keepData = false
): Promise<void> => {
  await api.delete(`/repositories/${name}`, { params: { keep_data: keepData } });
};

// Indexing
export const startIndexing = async (
  payload: IndexRequest
): Promise<IndexJobStatus> => {
  const res = await api.post<IndexJobStatus>('/index', payload);
  return res.data;
};

export const getIndexStatus = async (jobId: string): Promise<IndexJobStatus> => {
  const res = await api.get<IndexJobStatus>(`/index/${jobId}`);
  return res.data;
};

export const getStats = async (): Promise<StatsResponse> => {
  const res = await api.get<StatsResponse>('/stats');
  return res.data;
};

// Search
export const searchCode = async (payload: SearchRequest): Promise<SearchResponse> => {
  const res = await api.post<SearchResponse>('/search', payload);
  return res.data;
};

// Chat (Sync)
export const chatSync = async (payload: ChatRequest): Promise<ChatResponse> => {
  const res = await api.post<ChatResponse>('/chat/sync', payload);
  return res.data;
};

// Chat SSE Stream Handler
export interface SSECallbacks {
  onChunks?: (data: { results: RetrievedChunkOut[]; timings_ms: Record<string, number> }) => void;
  onToken?: (token: string) => void;
  onDone?: (data: {
    retrieval_time: number;
    generation_time: number;
    total_time: number;
    referenced_files?: string[];
    functions_used?: string[];
  }) => void;
  onError?: (err: string) => void;
}

export const streamChat = async (
  payload: ChatRequest,
  callbacks: SSECallbacks,
  signal?: AbortSignal
): Promise<void> => {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ message: 'Failed to initiate stream' }));
    callbacks.onError?.(errorData.message || `HTTP error ${response.status}`);
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError?.('ReadableStream not supported.');
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    let currentEvent = '';

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.replace('event: ', '').trim();
      } else if (line.startsWith('data: ')) {
        const rawData = line.replace('data: ', '').trim();
        try {
          const parsed = JSON.parse(rawData);
          if (currentEvent === 'chunks') {
            callbacks.onChunks?.(parsed);
          } else if (currentEvent === 'token') {
            callbacks.onToken?.(parsed.text);
          } else if (currentEvent === 'done') {
            callbacks.onDone?.(parsed);
          } else if (currentEvent === 'error') {
            callbacks.onError?.(parsed.message);
          }
        } catch {
          // Ignore malformed individual chunks
        }
      }
    }
  }
};

// Settings
export const getSettings = async (): Promise<SettingsResponse> => {
  const res = await api.get<SettingsResponse>('/settings');
  return res.data;
};

export const updateSettings = async (
  payload: SettingsUpdateRequest
): Promise<SettingsResponse> => {
  const res = await api.patch<SettingsResponse>('/settings', payload);
  return res.data;
};