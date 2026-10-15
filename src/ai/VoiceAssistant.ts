import { EventBus } from '../utils/EventBus';
import type { VoiceAssistantEvents } from './types';

// Browser vendor prefix types for SpeechRecognition
type SpeechRecognitionType = new () => {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onerror: ((event: { error: string; message?: string }) => void) | null;
  onresult: ((event: {
    resultIndex: number;
    results: Array<{
      isFinal: boolean;
      [index: number]: { transcript: string; confidence: number };
    }>;
  }) => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
};

export class VoiceAssistant {
  readonly events = new EventBus<VoiceAssistantEvents>();

  private recognition: InstanceType<SpeechRecognitionType> | null = null;
  private listening = false;
  private speaking = false;
  private muted = false;

  private voiceAmplitude = 0;
  private amplitudePhase = 0;
  private currentUtterance: SpeechSynthesisUtterance | null = null;
  private speechSupported = false;
  private recognitionSupported = false;

  constructor() {
    const win = window as unknown as {
      SpeechRecognition?: SpeechRecognitionType;
      webkitSpeechRecognition?: SpeechRecognitionType;
    };
    const RecognitionCtor = win.SpeechRecognition ?? win.webkitSpeechRecognition;

    if (RecognitionCtor) {
      this.recognitionSupported = true;
      try {
        this.recognition = new RecognitionCtor();
        this.recognition.continuous = false;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US';

        this.recognition.onstart = () => {
          this.listening = true;
          this.events.emit('listeningStart', undefined as void);
        };

        this.recognition.onend = () => {
          this.listening = false;
          this.events.emit('listeningEnd', undefined as void);
        };

        this.recognition.onerror = (e) => {
          this.listening = false;
          this.events.emit('error', `Speech recognition error: ${e.error}`);
        };

        this.recognition.onresult = (event) => {
          const results = event.results;
          let interim = '';
          let final = '';

          for (let i = event.resultIndex; i < results.length; i++) {
            const res = results[i];
            const text = res[0]?.transcript ?? '';
            if (res.isFinal) {
              final += text;
            } else {
              interim += text;
            }
          }

          if (final) {
            this.events.emit('transcript', { text: final.trim(), isFinal: true });
          } else if (interim) {
            this.events.emit('transcript', { text: interim.trim(), isFinal: false });
          }
        };
      } catch (err) {
        console.warn('[VoiceAssistant] failed to initialize SpeechRecognition:', err);
      }
    }

    this.speechSupported = typeof window !== 'undefined' && 'speechSynthesis' in window;
  }

  get isSupported(): boolean {
    return this.recognitionSupported || this.speechSupported;
  }

  get isListening(): boolean {
    return this.listening;
  }

  get isSpeaking(): boolean {
    return this.speaking;
  }

  get activeUtterance(): SpeechSynthesisUtterance | null {
    return this.currentUtterance;
  }

  get isMuted(): boolean {
    return this.muted;
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    if (muted && this.speaking) {
      this.stopSpeaking();
    }
  }

  startListening(): boolean {
    if (!this.recognition) {
      this.events.emit('error', 'Speech recognition not supported in this browser.');
      return false;
    }
    if (this.speaking) {
      this.stopSpeaking();
    }
    if (this.listening) return true;

    try {
      this.recognition.start();
      return true;
    } catch (err) {
      console.warn('[VoiceAssistant] startListening error:', err);
      return false;
    }
  }

  stopListening(): void {
    if (!this.recognition || !this.listening) return;
    try {
      this.recognition.stop();
    } catch {
      // Ignore
    }
    this.listening = false;
  }

  toggleListening(): boolean {
    if (this.listening) {
      this.stopListening();
      return false;
    } else {
      return this.startListening();
    }
  }

  speak(text: string): void {
    if (this.muted || !this.speechSupported || !text.trim()) return;

    this.stopSpeaking();

    // Strip markdown formatting for cleaner natural pronunciation
    const cleanText = text
      .replace(/\[Source:[^\]]+\]/g, '')
      .replace(/[*_#`~[\]]/g, '')
      .trim();

    if (!cleanText) return;

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;

    // Prefer high-quality modern English voice if available
    const voices = window.speechSynthesis.getVoices();
    const preferred = voices.find(
      (v) => (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha') || v.name.includes('Daniel')) && v.lang.startsWith('en')
    );
    if (preferred) utterance.voice = preferred;

    utterance.onstart = () => {
      this.speaking = true;
      this.events.emit('speakingStart', undefined as void);
    };

    utterance.onend = () => {
      this.speaking = false;
      this.voiceAmplitude = 0;
      this.currentUtterance = null;
      this.events.emit('speakingEnd', undefined as void);
    };

    utterance.onerror = () => {
      this.speaking = false;
      this.voiceAmplitude = 0;
      this.currentUtterance = null;
      this.events.emit('speakingEnd', undefined as void);
    };

    this.currentUtterance = utterance;
    window.speechSynthesis.speak(utterance);
  }

  stopSpeaking(): void {
    if (!this.speechSupported) return;
    window.speechSynthesis.cancel();
    this.speaking = false;
    this.voiceAmplitude = 0;
    this.currentUtterance = null;
    this.events.emit('speakingEnd', undefined as void);
  }

  /**
   * Per-frame update for phonetic cadence simulation.
   * Produces authentic voice amplitude envelope (0..1) to ripple the orb surface.
   */
  update(dt: number): number {
    if (this.speaking) {
      this.amplitudePhase += dt * 14.0;
      // Multi-harmonic phonetic envelope
      const env =
        0.5 +
        0.3 * Math.sin(this.amplitudePhase * 0.7) +
        0.15 * Math.sin(this.amplitudePhase * 2.1) +
        0.1 * Math.sin(this.amplitudePhase * 4.3);
      this.voiceAmplitude = Math.max(0.15, Math.min(1.0, env));
      this.events.emit('amplitude', this.voiceAmplitude);
    } else {
      this.voiceAmplitude = Math.max(0, this.voiceAmplitude - dt * 4.0);
    }
    return this.voiceAmplitude;
  }

  dispose(): void {
    this.stopListening();
    this.stopSpeaking();
    this.events.clear();
  }
}
