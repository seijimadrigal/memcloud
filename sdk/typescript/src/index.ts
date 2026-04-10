/**
 * Memcloud TypeScript SDK — Memory-as-a-service for AI agents
 *
 * Usage:
 *   import { MemChipClient } from 'memchip';
 *
 *   const mc = new MemChipClient({ apiKey: 'mc_xxx', userId: 'seiji' });
 *   await mc.add('User prefers dark theme');
 *   const results = await mc.search('theme preference');
 *   const answer = await mc.answer('What theme?');
 */

export interface MemChipConfig {
  apiKey: string;
  apiUrl?: string;
  userId?: string;
  agentId?: string;
  poolId?: string;
}

export interface Memory {
  id: string;
  content: string;
  memory_type: string;
  agent_id?: string;
  pool_id?: string;
  categories?: string[];
  structured_data?: Record<string, unknown>;
  confidence?: number;
  created_at: string;
  updated_at?: string;
}

export interface SearchResult {
  memories: Array<Memory & { rrf_score?: number; sources?: string[] }>;
  context: string;
  num_candidates: number;
  num_returned: number;
}

export interface AddResult {
  status: string;
  memories_created: number;
  counts: Record<string, number>;
  memory_ids: string[];
}

export class MemChipClient {
  private apiUrl: string;
  private apiKey: string;
  private userId: string;
  private agentId?: string;
  private poolId?: string;

  constructor(config: MemChipConfig) {
    this.apiUrl = (config.apiUrl || 'https://api.memcloud.dev/v1').replace(/\/$/, '');
    this.apiKey = config.apiKey;
    this.userId = config.userId || 'default';
    this.agentId = config.agentId;
    this.poolId = config.poolId;
  }

  private async request(path: string, options: RequestInit = {}): Promise<any> {
    const url = `${this.apiUrl}${path}`;
    const res = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
        ...options.headers,
      },
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Memcloud API ${res.status}: ${text}`);
    }
    return res.json();
  }

  async add(text: string, opts?: { agentId?: string; poolId?: string; sessionId?: string; metadata?: Record<string, unknown> }): Promise<AddResult> {
    return this.request('/memories/', {
      method: 'POST',
      body: JSON.stringify({
        text,
        user_id: this.userId,
        agent_id: opts?.agentId || this.agentId,
        pool_id: opts?.poolId || this.poolId,
        session_id: opts?.sessionId,
        metadata: opts?.metadata,
      }),
    });
  }

  async search(query: string, opts?: { topK?: number; poolId?: string; searchScope?: string[]; agentic?: boolean }): Promise<SearchResult> {
    return this.request('/memories/search/', {
      method: 'POST',
      body: JSON.stringify({
        query,
        user_id: this.userId,
        agent_id: this.agentId,
        pool_id: opts?.poolId || this.poolId,
        search_scope: opts?.searchScope,
        top_k: opts?.topK || 5,
        agentic: opts?.agentic ?? true,
      }),
    });
  }

  async answer(question: string): Promise<string> {
    const res = await this.request('/memories/answer/', {
      method: 'POST',
      body: JSON.stringify({
        question,
        user_id: this.userId,
        agent_id: this.agentId,
      }),
    });
    return res.answer || '';
  }

  async recall(opts?: {
    query?: string;
    agentId?: string;
    tokenBudget?: number;
    format?: 'xml' | 'markdown' | 'text';
    includeProfile?: boolean;
    includeRecent?: boolean;
    topK?: number;
  }): Promise<{
    context: string;
    format: string;
    token_count: number;
    sections: Record<string, number>;
    latency_ms: number;
  }> {
    return this.request('/recall', {
      method: 'POST',
      body: JSON.stringify({
        user_id: this.userId,
        agent_id: opts?.agentId || this.agentId,
        query: opts?.query,
        token_budget: opts?.tokenBudget || 4000,
        format: opts?.format || 'markdown',
        include_profile: opts?.includeProfile ?? true,
        include_recent: opts?.includeRecent ?? true,
        top_k: opts?.topK || 15,
      }),
    });
  }

  async list(opts?: { memoryType?: string; poolId?: string; limit?: number; offset?: number }): Promise<Memory[]> {
    const params = new URLSearchParams({
      user_id: this.userId,
      limit: String(opts?.limit || 50),
      offset: String(opts?.offset || 0),
    });
    if (opts?.memoryType) params.set('memory_type', opts.memoryType);
    if (opts?.poolId || this.poolId) params.set('pool_id', opts?.poolId || this.poolId!);
    return this.request(`/memories/?${params}`);
  }

  async get(memoryId: string): Promise<Memory> {
    return this.request(`/memories/${memoryId}`);
  }

  async update(memoryId: string, opts: { content?: string; metadata?: Record<string, unknown> }): Promise<Memory> {
    return this.request(`/memories/${memoryId}`, {
      method: 'PUT',
      body: JSON.stringify(opts),
    });
  }

  async delete(memoryId: string): Promise<void> {
    await this.request(`/memories/${memoryId}`, { method: 'DELETE' });
  }

  async bulkImport(memories: Array<{ content: string; memory_type?: string; agent_id?: string; pool_id?: string; categories?: string[] }>): Promise<{ imported: number }> {
    return this.request('/memories/bulk/import/', {
      method: 'POST',
      body: JSON.stringify({ user_id: this.userId, memories }),
    });
  }

  async bulkExport(opts?: { memoryType?: string; poolId?: string; limit?: number }): Promise<{ count: number; memories: Memory[] }> {
    return this.request('/memories/bulk/export/', {
      method: 'POST',
      body: JSON.stringify({
        user_id: this.userId,
        memory_type: opts?.memoryType,
        pool_id: opts?.poolId,
        limit: opts?.limit || 1000,
      }),
    });
  }

  async stats(): Promise<Record<string, unknown>> {
    return this.request('/stats/');
  }

  async analytics(days = 30): Promise<Record<string, unknown>> {
    return this.request(`/analytics/?days=${days}`);
  }

  // Sessions
  async createSession(opts?: { name?: string; expiresInMinutes?: number }): Promise<Record<string, unknown>> {
    return this.request('/sessions/', {
      method: 'POST',
      body: JSON.stringify({
        user_id: this.userId,
        agent_id: this.agentId,
        name: opts?.name,
        expires_in_minutes: opts?.expiresInMinutes,
      }),
    });
  }

  // Webhooks
  async createWebhook(url: string, events: string[], poolFilter?: string): Promise<Record<string, unknown>> {
    return this.request('/webhooks/', {
      method: 'POST',
      body: JSON.stringify({ url, events, pool_filter: poolFilter }),
    });
  }

  async listWebhooks(): Promise<Array<Record<string, unknown>>> {
    return this.request('/webhooks/');
  }
}
