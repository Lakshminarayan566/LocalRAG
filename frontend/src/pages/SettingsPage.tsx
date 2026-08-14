import React, { useState, useEffect } from 'react';
import { getSettings, updateSettings } from '../services/api';
import { SettingsResponse } from '../types/api';
import { Save, Sliders, CheckCircle2, AlertCircle } from 'lucide-react';

export const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<SettingsResponse | null>(null);
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getSettings()
      .then(setSettings)
      .catch((err) => setError('Failed to load backend settings'));
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!settings) return;

    try {
      setSaving(true);
      setError(null);
      setSuccess(false);

      const updated = await updateSettings({
        llm_model: settings.llm_model,
        llm_temperature: settings.llm_temperature,
        top_k_vector: settings.top_k_vector,
        top_k_bm25: settings.top_k_bm25,
        final_top_k: settings.final_top_k,
        use_reranker: settings.use_reranker,
      });

      setSettings(updated);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to update settings');
    } finally {
      setSaving(false);
    }
  };

  if (!settings) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-400 text-xs">
        Loading settings...
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-8 max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white mb-1">Configuration</h1>
        <p className="text-xs text-gray-400">
          Tune runtime parameters for Ollama LLM, retrieval weights, and cross-encoder reranking.
        </p>
      </div>

      {success && (
        <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-xl flex items-center space-x-2 text-emerald-400 text-xs">
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          <span>Settings successfully updated. Changes will take effect on the next query.</span>
        </div>
      )}

      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl flex items-center space-x-2 text-red-400 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <form onSubmit={handleSave} className="space-y-6">
        {/* LLM Settings */}
        <div className="bg-surface border border-border p-6 rounded-2xl space-y-4">
          <h2 className="text-sm font-semibold text-white flex items-center space-x-2">
            <Sliders className="w-4 h-4 text-accent-light" />
            <span>Ollama Generation Parameters</span>
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Ollama Model</label>
              <input
                type="text"
                value={settings.llm_model}
                onChange={(e) => setSettings({ ...settings, llm_model: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-accent"
              />
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Temperature ({settings.llm_temperature})</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.llm_temperature}
                onChange={(e) => setSettings({ ...settings, llm_temperature: parseFloat(e.target.value) })}
                className="w-full accent-accent cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* Retrieval Settings */}
        <div className="bg-surface border border-border p-6 rounded-2xl space-y-4">
          <h2 className="text-sm font-semibold text-white">Hybrid Retrieval Parameters</h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Top-K Vector Candidates</label>
              <input
                type="number"
                value={settings.top_k_vector}
                onChange={(e) => setSettings({ ...settings, top_k_vector: parseInt(e.target.value) || 1 })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-accent"
              />
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Top-K BM25 Candidates</label>
              <input
                type="number"
                value={settings.top_k_bm25}
                onChange={(e) => setSettings({ ...settings, top_k_bm25: parseInt(e.target.value) || 1 })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-accent"
              />
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Final Top-K Chunks</label>
              <input
                type="number"
                value={settings.final_top_k}
                onChange={(e) => setSettings({ ...settings, final_top_k: parseInt(e.target.value) || 1 })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-accent"
              />
            </div>
          </div>

          <div className="flex items-center space-x-2 pt-2">
            <input
              type="checkbox"
              id="use_reranker"
              checked={settings.use_reranker}
              onChange={(e) => setSettings({ ...settings, use_reranker: e.target.checked })}
              className="accent-accent rounded cursor-pointer"
            />
            <label htmlFor="use_reranker" className="text-xs text-gray-300 cursor-pointer">
              Enable Cross-Encoder Reranker
            </label>
          </div>
        </div>

        <button
          type="submit"
          disabled={saving}
          className="bg-accent hover:bg-accent-hover text-white text-xs font-medium px-5 py-2.5 rounded-xl transition-colors flex items-center space-x-2 shadow-lg shadow-accent/20"
        >
          <Save className="w-4 h-4" />
          <span>{saving ? 'Saving...' : 'Save Settings'}</span>
        </button>
      </form>
    </div>
  );
};