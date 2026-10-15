import { EventBus } from '../utils/EventBus';
import type { ChatMessage, Citation, ToolCall } from './types';

interface ClientEvents extends Record<string, unknown> {
  token: string;
  toolCall: ToolCall;
  citation: Citation[];
  done: { fullText: string; ttftMs: number; totalMs: number };
  error: string;
}

export class AIClient {
  readonly events = new EventBus<ClientEvents>();
  private readonly messages: ChatMessage[] = [];
  private abortController: AbortController | null = null;
  private busy = false;

  // Latency telemetry
  private lastTtftMs = 0;
  private lastTotalMs = 0;

  get isBusy(): boolean {
    return this.busy;
  }

  get history(): readonly ChatMessage[] {
    return this.messages;
  }

  get ttftMs(): number {
    return this.lastTtftMs;
  }

  get totalMs(): number {
    return this.lastTotalMs;
  }

  async sendMessage(userText: string): Promise<void> {
    if (this.busy) {
      this.abort();
    }

    const trimmed = userText.trim();
    if (!trimmed) return;

    this.busy = true;
    this.abortController = new AbortController();

    const userMessage: ChatMessage = {
      id: `usr_${Date.now()}`,
      role: 'user',
      content: trimmed,
      timestamp: Date.now(),
    };
    this.messages.push(userMessage);

    const startedAt = performance.now();
    let firstTokenAt = 0;
    let fullReply = '';
    const collectedToolCalls: ToolCall[] = [];
    const collectedCitations: Citation[] = [];

    const payload = {
      message: trimmed,
      history: this.messages.slice(-8).map((m) => ({
        role: m.role,
        content: m.content,
        timestamp: m.timestamp / 1000,
      })),
    };

    try {
      let response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: this.abortController.signal,
      });

      if (response.status === 404) {
        // Transparent fallback to /api/chat/stream with universal payload
        const fallbackPayload = {
          message: trimmed,
          messages: [
            ...this.messages.slice(-8).map((m) => ({ role: m.role, content: m.content })),
            { role: 'user', content: trimmed },
          ],
        };
        response = await fetch('/api/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(fallbackPayload),
          signal: this.abortController.signal,
        });
      }

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Backend response ${response.status}: ${errorText || response.statusText}`);
      }

      if (!response.body) {
        throw new Error('Response body is null.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine.startsWith('data: ')) continue;
          const jsonStr = trimmedLine.slice(6).trim();
          if (jsonStr === '[DONE]') continue;

          try {
            const event = JSON.parse(jsonStr) as Record<string, unknown>;

            // Support both v2 format {type: 'token', content: '...'} and legacy {token: '...'}
            const rawToken = event.token ?? (event.type === 'token' ? (event.content ?? event.data) : null);
            if (rawToken !== null && rawToken !== undefined) {
              const token = String(rawToken);
              if (!firstTokenAt) {
                firstTokenAt = performance.now();
                this.lastTtftMs = Math.round(firstTokenAt - startedAt);
              }
              fullReply += token;
              this.events.emit('token', token);
            } else if (event.type === 'tool_call') {
              const tc: ToolCall = {
                id: String(event.id ?? Date.now()),
                name: String(event.name ?? (event.data as Record<string, unknown>)?.name ?? ''),
                arguments: (event.arguments ?? (event.data as Record<string, unknown>)?.arguments ?? {}) as Record<string, unknown>,
              };
              collectedToolCalls.push(tc);
              this.events.emit('toolCall', tc);
            } else if (event.type === 'citation') {
              const citations = (event.citations ?? event.data ?? []) as Citation[];
              collectedCitations.push(...citations);
              this.events.emit('citation', citations);
            } else if (event.type === 'error' || event.error) {
              const err = String(event.message ?? event.error ?? event.data ?? 'Provider error');
              this.events.emit('error', err);
            } else if (event.type === 'done') {
              // Stream terminal
            }
          } catch {
            // Partial JSON packet in line, ignore
          }
        }
      }

      this.lastTotalMs = Math.round(performance.now() - startedAt);
      if (!firstTokenAt) this.lastTtftMs = this.lastTotalMs;

      const assistantMessage: ChatMessage = {
        id: `asst_${Date.now()}`,
        role: 'assistant',
        content: fullReply,
        timestamp: Date.now(),
        toolCalls: collectedToolCalls.length ? collectedToolCalls : undefined,
        citations: collectedCitations.length ? collectedCitations : undefined,
      };
      this.messages.push(assistantMessage);

      this.events.emit('done', {
        fullText: fullReply,
        ttftMs: this.lastTtftMs,
        totalMs: this.lastTotalMs,
      });
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        this.events.emit('error', 'Request was interrupted.');
      } else {
        const msg = err instanceof Error ? err.message : String(err);
        this.events.emit('error', msg);
      }
    } finally {
      this.busy = false;
      this.abortController = null;
    }
  }

  abort(): void {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
    this.busy = false;
  }

  clear(): void {
    void this.clearHistory();
  }

  async clearHistory(): Promise<void> {
    this.messages.length = 0;
    try {
      await fetch('/api/chat/clear', { method: 'POST' });
    } catch {
      // Local clear is sufficient even if offline
    }
  }
}
