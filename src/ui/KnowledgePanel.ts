import { el } from './dom';

export interface DocumentInfo {
  id: string;
  filename: string;
  chunks_count: number;
  total_characters: number;
}

export class KnowledgePanel {
  readonly root: HTMLDivElement;
  private readonly docListContainer: HTMLDivElement;
  private readonly fileInput: HTMLInputElement;
  private readonly statusText: HTMLDivElement;
  private open = false;

  constructor(parent: HTMLElement) {
    this.root = el('div', 'panel hud-knowledge');

    // Header
    const header = el('div', 'chat-header');
    const title = el('span', 'chat-title', 'DOCUMENT KNOWLEDGE VAULT');
    const closeBtn = el('button', 'chat-toggle-btn', '—');
    closeBtn.addEventListener('click', () => this.toggle());
    header.append(title, closeBtn);

    const body = el('div', 'knowledge-body');

    // Upload box
    const uploadArea = el('div', 'upload-area');
    const uploadLabel = el('span', 'upload-prompt', 'DROP TXT / PDF OR CLICK TO UPLOAD');
    this.fileInput = el('input', 'file-input');
    this.fileInput.type = 'file';
    this.fileInput.accept = '.txt,.pdf,.md,.json';
    this.fileInput.addEventListener('change', () => this.handleFileSelect());

    uploadArea.append(uploadLabel, this.fileInput);

    this.statusText = el('div', 'upload-status', 'READY');

    // Document list
    const listHeader = el('div', 'knowledge-section-title', 'INDEXED DOCUMENTS:');
    this.docListContainer = el('div', 'doc-list');

    const clearBtn = el('button', 'chat-btn clear-vault', 'CLEAR ALL DOCUMENTS');
    clearBtn.addEventListener('click', () => void this.clearAll());

    body.append(uploadArea, this.statusText, listHeader, this.docListContainer, clearBtn);
    this.root.append(header, body);
    parent.append(this.root);

    // Initial load
    void this.refreshDocuments();
  }

  get isOpen(): boolean {
    return this.open;
  }

  setOpen(open: boolean): void {
    this.open = open;
    this.root.classList.toggle('open', open);
    if (open) void this.refreshDocuments();
  }

  toggle(): boolean {
    this.setOpen(!this.open);
    return this.open;
  }

  private async handleFileSelect(): Promise<void> {
    const file = this.fileInput.files?.[0];
    if (!file) return;

    this.statusText.textContent = `UPLOADING & INDEXING ${file.name}...`;
    this.statusText.className = 'upload-status active';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await res.json();
      this.statusText.textContent = `INDEXED: ${data.chunks_count} chunks (${data.total_characters} chars)`;
      this.statusText.className = 'upload-status success';
      this.fileInput.value = '';
      await this.refreshDocuments();
    } catch (e) {
      this.statusText.textContent = `ERROR: ${e instanceof Error ? e.message : String(e)}`;
      this.statusText.className = 'upload-status error';
    }
  }

  async refreshDocuments(): Promise<void> {
    try {
      const res = await fetch('/api/documents');
      if (!res.ok) return;
      const docs = (await res.json()) as DocumentInfo[];

      this.docListContainer.replaceChildren();

      if (!docs.length) {
        const empty = el('div', 'doc-empty', 'NO DOCUMENTS LOADED');
        this.docListContainer.append(empty);
        return;
      }

      for (const d of docs) {
        const item = el('div', 'doc-item');
        const info = el('div', 'doc-info');
        const name = el('span', 'doc-name', d.filename);
        const meta = el('span', 'doc-meta', `${d.chunks_count} chunks · ${d.total_characters.toLocaleString()} chars`);
        info.append(name, meta);

        const delBtn = el('button', 'doc-del-btn', '✕');
        delBtn.title = 'Remove Document';
        delBtn.addEventListener('click', async () => {
          await fetch(`/api/documents/${d.id}`, { method: 'DELETE' });
          await this.refreshDocuments();
        });

        item.append(info, delBtn);
        this.docListContainer.append(item);
      }
    } catch {
      // Backend not running yet
    }
  }

  async clearAll(): Promise<void> {
    try {
      await fetch('/api/documents/clear', { method: 'POST' });
      this.statusText.textContent = 'VAULT CLEARED';
      this.statusText.className = 'upload-status';
      await this.refreshDocuments();
    } catch (e) {
      this.statusText.textContent = `CLEAR FAILED: ${String(e)}`;
    }
  }
}
