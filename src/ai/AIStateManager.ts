import { EventBus } from '../utils/EventBus';
import type { AIState } from './types';

interface StateEvents extends Record<string, unknown> {
  change: { from: AIState; to: AIState };
  timeout: { state: AIState };
}

export interface StateVisualProfile {
  energyTarget: number;
  pulseRate: number;      // Multiplier on pulse frequency
  breathRate: number;     // Amplitude of breathing motion
  warmthTarget: number;   // 0 = cyan, 1 = golden
  alert: boolean;         // Triggers post-processing alert wash
}

export const STATE_PROFILES: Record<AIState, StateVisualProfile> = {
  IDLE: {
    energyTarget: 0.22,
    pulseRate: 1.0,
    breathRate: 1.0,
    warmthTarget: 0.0,
    alert: false,
  },
  LISTENING: {
    energyTarget: 0.65,
    pulseRate: 1.5,
    breathRate: 1.6,
    warmthTarget: 0.0,
    alert: false,
  },
  THINKING: {
    energyTarget: 0.88,
    pulseRate: 3.2,
    breathRate: 2.2,
    warmthTarget: 0.2,
    alert: false,
  },
  SPEAKING: {
    energyTarget: 0.95,
    pulseRate: 2.0,
    breathRate: 1.8,
    warmthTarget: 0.35,
    alert: false,
  },
  SCANNING: {
    energyTarget: 0.82,
    pulseRate: 1.8,
    breathRate: 1.2,
    warmthTarget: 0.1,
    alert: false,
  },
  ERROR: {
    energyTarget: 0.45,
    pulseRate: 0.5,
    breathRate: 0.4,
    warmthTarget: 0.85,
    alert: true,
  },
};

export class AIStateManager {
  readonly events = new EventBus<StateEvents>();
  private current: AIState = 'IDLE';
  private timer = 0;
  private readonly stateTimeouts: Partial<Record<AIState, number>> = {
    LISTENING: 25.0,  // Max listening time without speech
    THINKING: 45.0,   // Max wait for LLM response
    ERROR: 6.0,       // Auto-clear error after 6s
  };

  get state(): AIState {
    return this.current;
  }

  get profile(): StateVisualProfile {
    return STATE_PROFILES[this.current];
  }

  setState(next: AIState): void {
    if (this.current === next) return;
    const previous = this.current;
    this.current = next;
    this.timer = 0;
    this.events.emit('change', { from: previous, to: next });
  }

  update(dt: number): void {
    this.timer += dt;
    const timeout = this.stateTimeouts[this.current];
    if (timeout && this.timer >= timeout) {
      const expired = this.current;
      this.events.emit('timeout', { state: expired });
      this.setState('IDLE');
    }
  }

  reset(): void {
    this.setState('IDLE');
  }
}
