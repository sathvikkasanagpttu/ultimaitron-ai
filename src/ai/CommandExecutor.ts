import type { ToolCall } from './types';
import type { OrbMode } from '../orb/OrbModes';

export interface CommandTargets {
  setMode: (mode: OrbMode) => void;
  firePulse: (intensity?: number) => void;
  triggerBurst: (warm?: number) => void;
  setScale: (scale: number) => void;
  setWarmth: (warmth: number) => void;
  resetSystem: () => void;
  openPanel: (panel: 'chat' | 'diagnostics' | 'knowledge') => void;
}

export class CommandExecutor {
  constructor(private readonly targets: CommandTargets) {}

  execute(toolCall: ToolCall): { success: boolean; message: string } {
    const { name, arguments: args } = toolCall;

    try {
      switch (name) {
        case 'change_orb_mode': {
          const mode = String(args.mode).toUpperCase() as OrbMode;
          if (['NEURAL', 'REACTOR', 'DATA', 'ENERGY', 'SCANNER'].includes(mode)) {
            this.targets.setMode(mode);
            return { success: true, message: `Configuration morphed to ${mode}` };
          }
          return { success: false, message: `Unknown mode ${mode}` };
        }

        case 'fire_pulse': {
          const intensity = Number(args.intensity ?? 1.0);
          this.targets.firePulse(intensity);
          return { success: true, message: `Energy pulse fired (intensity: ${intensity.toFixed(1)})` };
        }

        case 'trigger_burst': {
          const warm = Number(args.warm ?? 0.0);
          this.targets.triggerBurst(warm);
          return { success: true, message: `Shockwave burst triggered (warm: ${warm.toFixed(1)})` };
        }

        case 'set_orb_scale': {
          const scale = Number(args.scale ?? 1.0);
          const safeScale = Math.max(0.62, Math.min(1.85, scale));
          this.targets.setScale(safeScale);
          return { success: true, message: `Orb scale adjusted to ${safeScale.toFixed(2)}x` };
        }

        case 'set_warmth': {
          const warmth = Number(args.warmth ?? 0.0);
          const safeWarm = Math.max(0.0, Math.min(1.0, warmth));
          this.targets.setWarmth(safeWarm);
          return { success: true, message: `Spectrum shifted (warmth: ${safeWarm.toFixed(2)})` };
        }

        case 'open_panel': {
          const panel = String(args.panel).toLowerCase() as 'chat' | 'diagnostics' | 'knowledge';
          if (['chat', 'diagnostics', 'knowledge'].includes(panel)) {
            this.targets.openPanel(panel);
            return { success: true, message: `Opened ${panel} panel` };
          }
          return { success: false, message: `Unknown panel: ${panel}` };
        }

        case 'reset_system': {
          this.targets.resetSystem();
          return { success: true, message: 'System reset to default state' };
        }

        default:
          return { success: false, message: `Command '${name}' rejected by security policy.` };
      }
    } catch (err) {
      return { success: false, message: `Execution failed: ${String(err)}` };
    }
  }
}
