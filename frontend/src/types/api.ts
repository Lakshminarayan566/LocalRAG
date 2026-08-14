export interface ErrorResponse {
  error: string;
  message: string;
  detail?: string;
}

export interface HealthResponse {
  status: 'ok' | 'degraded';
  ollama_reachable: boolean;
  ollama_model: string;
  active_repository: string | null;
  indexed_chunks: number | null;
}

export interface RepositoryAddRequest {
  path: string;
  name?: string;
}

export interface RepositoryResponse {
  name: string;
  path: string;
  collection: string;
  bm25_dir: string;
  is_active: boolean;
}

export interface RepositoryListResponse {
  active: string | null;
  repositories: RepositoryResponse[];
}

export interface IndexRequest {
  repo?: string;
  languages?: string[];
}

export interface IndexJobStatus {
  job_id: string;
  status: 'running' | 'completed' | 'failed';
  repo?: string;
  result?: Record<string, any>;
  error?: string;
}

export interface StatsResponse {
  total_chunks: number;
  collection_name?: string;
  persist_dir?: string;
  embedding_model?: string;
  unique_files?: number;
  languages?: Record<string, number>;
  chunk_types?: Record<string, number>;
  embedding_dimension?: number;
  bm25_index_size?: number;
  bm25_index_built?: boolean;
}

export interface SearchRequest {
  query: string;
  language?: string;
  chunk_type?: string;
  repo?: string;
}

export interface RetrievedChunkOut {
  chunk_id: string;
  file_path: string;
  function_name: string;
  class_name: string;
  language: string;
  chunk_type: string;
  start_line: number;
  end_line: number;
  code_content: string;
  vector_score: number;
  bm25_score: number;
  rrf_score: number;
  rerank_score: number;
  final_rank: number;
}

export interface SearchResponse {
  query: string;
  results: RetrievedChunkOut[];
  timings_ms: Record<string, number>;
}

export type TaskType =
  | 'general'
  | 'explain'
  | 'find_bugs'
  | 'similar_code'
  | 'function_search'
  | 'class_search';

export interface ChatRequest {
  query: string;
  task_focus?: TaskType;
  repo_name?: string;

  language?: string;
  chunk_type?: string;
  file_path?: string;
  class_name?: string;

  temperature?: number;
  top_k?: number;
  use_reranker?: boolean;
}

export interface ChatResponse {
  answer: string;
  retrieved_chunks: RetrievedChunkOut[];
  referenced_files: string[];
  functions_used: string[];
  timings_ms: Record<string, number>;
}

export interface SettingsResponse {
  embedding_model: string;
  collection: string;
  persist_dir: string;
  llm_model: string;
  llm_max_tokens: number;
  llm_temperature: number;
  llm_keep_alive: string;
  top_k_vector: number;
  top_k_bm25: number;
  rrf_k: number;
  rerank_candidates: number;
  final_top_k: number;
  use_reranker: boolean;
}

export interface SettingsUpdateRequest {
  llm_model?: string;
  llm_temperature?: number;
  llm_max_tokens?: number;
  llm_context_window?: number;
  llm_keep_alive?: string;
  top_k_vector?: number;
  top_k_bm25?: number;
  rerank_candidates?: number;
  final_top_k?: number;
  use_reranker?: boolean;
}