/**
 * Type declarations for the Ultimaitron AI 2.0 companion layer.
 */

export type AIState = 'IDLE' | 'LISTENING' | 'THINKING' | 'SPEAKING' | 'SCANNING' | 'ERROR';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  toolCalls?: ToolCall[];
  citations?: Citation[];
}

export interface Citation {
  filename: string;
  chunk_index: number;
  score: number;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface AIClientEvents extends Record<string, unknown> {
  state: AIState;
  token: string;
  citation: Citation[];
  tool_call: ToolCall;
  done: void;
  error: string;
  metrics: {
    ttftMs: number;
    totalMs: number;
  };
}

export interface VoiceAssistantEvents extends Record<string, unknown> {
  listeningStart: void;
  listeningEnd: void;
  transcript: { text: string; isFinal: boolean };
  speakingStart: void;
  speakingEnd: void;
  amplitude: number; // 0.0 to 1.0 real-time voice amplitude
  error: string;
}

