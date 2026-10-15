import { el } from './dom';
import type { Citation, ToolCall } from '../ai/types';

export interface ChatPanelCallbacks {
  onSendMessage: (text: string) => void;
  onToggleVoice: () => void;
  onClearChat: () => void;
}

export class ChatPanel {
  readonly root: HTMLDivElement;

  private readonly messageContainer: HTMLDivElement;
  private readonly input: HTMLInputElement;
  private readonly sendBtn: HTMLButtonElement;
  private readonly micBtn: HTMLButtonElement;
  private readonly statusIndicator: HTMLSpanElement;
  private currentStreamNode: HTMLDivElement | null = null;
  private open = false;

  constructor(parent: HTMLElement, private readonly callbacks: ChatPanelCallbacks) {
    this.root = el('div', 'panel hud-chat');

    // Header
    const header = el('div', 'chat-header');
    const title = el('span', 'chat-title', 'AI NEURAL TRANSCRIPT');
    this.statusIndicator = el('span', 'chat-state-tag', 'STANDBY');
    const closeBtn = el('button', 'chat-toggle-btn', '—');
    closeBtn.title = 'Minimize Transcript';
    closeBtn.addEventListener('click', () => this.toggle());

    header.append(title, this.statusIndicator, closeBtn);

    // Message history container
    this.messageContainer = el('div', 'chat-messages');

    // Input toolbar
    const form = el('form', 'chat-form');
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      this.submit();
    });

    this.input = el('input', 'chat-input');
    this.input.type = 'text';
    this.input.placeholder = 'Query AI core or speak...';
    this.input.maxLength = 1000;

    this.sendBtn = el('button', 'chat-btn send', 'SEND');
    this.sendBtn.type = 'submit';

    this.micBtn = el('button', 'chat-btn mic', 'MIC');
    this.micBtn.type = 'button';
    this.micBtn.title = 'Toggle Voice Recognition';
    this.micBtn.addEventListener('click', () => this.callbacks.onToggleVoice());

    const clearBtn = el('button', 'chat-btn clear', 'CLR');
    clearBtn.type = 'button';
    clearBtn.title = 'Clear Conversation';
    clearBtn.addEventListener('click', () => {
      this.clearMessages();
      this.callbacks.onClearChat();
    });

    form.append(this.input, this.sendBtn, this.micBtn, clearBtn);

    this.root.append(header, this.messageContainer, form);
    parent.append(this.root);
  }

  get isOpen(): boolean {
    return this.open;
  }

  setOpen(open: boolean): void {
    this.open = open;
    this.root.classList.toggle('open', open);
    if (open) {
      this.input.focus();
      this.scrollToBottom();
    }
  }

  toggle(): boolean {
    this.setOpen(!this.open);
    return this.open;
  }

  setState(stateText: string, level: 'normal' | 'warn' | 'active' = 'normal'): void {
    this.statusIndicator.textContent = stateText;
    this.statusIndicator.className = `chat-state-tag ${level}`;
  }

  setMicActive(active: boolean): void {
    this.micBtn.classList.toggle('active', active);
    this.micBtn.textContent = active ? 'REC' : 'MIC';
  }

  setInputValue(text: string): void {
    this.input.value = text;
  }

  appendUserMessage(text: string): void {
    const row = el('div', 'chat-msg user');
    const badge = el('span', 'role-badge', 'YOU');
    const content = el('div', 'msg-body', text);
    row.append(badge, content);
    this.messageContainer.append(row);
    this.scrollToBottom();
  }

  beginStream(): void {
    const row = el('div', 'chat-msg assistant streaming');
    const badge = el('span', 'role-badge', 'CORE');
    const content = el('div', 'msg-body', '');
    row.append(badge, content);
    this.messageContainer.append(row);
    this.currentStreamNode = content;
    this.scrollToBottom();
  }

  appendToken(token: string): void {
    if (!this.currentStreamNode) {
      this.beginStream();
    }
    if (this.currentStreamNode) {
      this.currentStreamNode.textContent += token;
      this.scrollToBottom();
    }
  }

  appendToolCall(toolCall: ToolCall): void {
    const pill = el('div', 'tool-call-pill');
    pill.textContent = `⚡ [ACTION: ${toolCall.name}]`;
    this.messageContainer.append(pill);
    this.scrollToBottom();
  }

  appendCitations(citations: Citation[]): void {
    if (!citations.length) return;
    const box = el('div', 'citation-box');
    const title = el('span', 'citation-title', 'RETRIEVED CITATIONS:');
    box.append(title);
    for (const c of citations) {
      const cite = el('span', 'citation-item', `• ${c.filename} (Chunk #${c.chunk_index})`);
      box.append(cite);
    }
    this.messageContainer.append(box);
    this.scrollToBottom();
  }

  endStream(): void {
    if (this.currentStreamNode) {
      this.currentStreamNode.parentElement?.classList.remove('streaming');
      this.currentStreamNode = null;
    }
  }

  appendError(errorText: string): void {
    const row = el('div', 'chat-msg error');
    const badge = el('span', 'role-badge', 'ERR');
    const content = el('div', 'msg-body', errorText);
    row.append(badge, content);
    this.messageContainer.append(row);
    this.scrollToBottom();
  }

  clearMessages(): void {
    this.messageContainer.replaceChildren();
    this.currentStreamNode = null;
  }

  private submit(): void {
    const text = this.input.value.trim();
    if (!text) return;
    this.input.value = '';
    this.appendUserMessage(text);
    this.callbacks.onSendMessage(text);
  }

  private scrollToBottom(): void {
    this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
  }
}
