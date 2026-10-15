import { el, row, divider, type RowHandle } from './dom';

export interface DiagnosticsTelemetry {
  fps: number;
  frameTimeMs: number;
  gestureLatencyMs: number;
  inferenceFps: number;
  ttftMs: number;
  aiResponseMs: number;
  aiState: string;
  cameraState: string;
  micState: string;
  backendConnected: boolean;
  qualityTier: string;
  particles: number;
  connections: number;
  drawCalls: number;
  errorsCount: number;
}

export class DiagnosticsPanel {
  readonly root: HTMLDivElement;

  private readonly fpsRow: RowHandle;
  private readonly frameTimeRow: RowHandle;
  private readonly gestureLatencyRow: RowHandle;
  private readonly inferenceFpsRow: RowHandle;
  private readonly ttftRow: RowHandle;
  private readonly aiTotalRow: RowHandle;
  private readonly aiStateRow: RowHandle;
  private readonly cameraRow: RowHandle;
  private readonly micRow: RowHandle;
  private readonly backendRow: RowHandle;
  private readonly qualityRow: RowHandle;
  private readonly particlesRow: RowHandle;
  private readonly drawCallsRow: RowHandle;
  private readonly errorsRow: RowHandle;

  private open = false;

  constructor(parent: HTMLElement) {
    this.root = el('div', 'panel hud-diagnostics');

    // Header
    const header = el('div', 'chat-header');
    const title = el('span', 'chat-title', 'DIAGNOSTICS & SYSTEM TELEMETRY');
    const closeBtn = el('button', 'chat-toggle-btn', '—');
    closeBtn.addEventListener('click', () => this.toggle());
    header.append(title, closeBtn);

    const body = el('div', 'diagnostics-body');

    // Rows
    this.fpsRow = row('RENDER FPS', '—');
    this.frameTimeRow = row('FRAME TIME', '—');
    this.gestureLatencyRow = row('GESTURE LATENCY', '—');
    this.inferenceFpsRow = row('INFERENCE RATE', '—');

    this.ttftRow = row('AI TIME TO FIRST TOKEN', '—');
    this.aiTotalRow = row('AI RESPONSE DURATION', '—');
    this.aiStateRow = row('HOLOGRAM AI STATE', 'IDLE');

    this.cameraRow = row('OPTICAL SENSOR', 'STANDBY');
    this.micRow = row('ACOUSTIC SENSOR', 'STANDBY');
    this.backendRow = row('BACKEND API STATUS', 'CHECKING...');

    this.qualityRow = row('QUALITY TIER', 'HIGH');
    this.particlesRow = row('ACTIVE PARTICLES', '—');
    this.drawCallsRow = row('GPU DRAW CALLS', '—');
    this.errorsRow = row('SYSTEM ERROR COUNT', '0');

    body.append(
      this.fpsRow.root,
      this.frameTimeRow.root,
      this.gestureLatencyRow.root,
      this.inferenceFpsRow.root,
      divider(),
      this.ttftRow.root,
      this.aiTotalRow.root,
      this.aiStateRow.root,
      divider(),
      this.cameraRow.root,
      this.micRow.root,
      this.backendRow.root,
      divider(),
      this.qualityRow.root,
      this.particlesRow.root,
      this.drawCallsRow.root,
      this.errorsRow.root
    );

    const honestNotice = el('div', 'diag-note', '* ALL TELEMETRY REPRESENTS ACTUAL RUNTIME BENCHMARKS (NO FABRICATED METRICS)');
    body.append(honestNotice);

    this.root.append(header, body);
    parent.append(this.root);
  }

  get isOpen(): boolean {
    return this.open;
  }

  setOpen(open: boolean): void {
    this.open = open;
    this.root.classList.toggle('open', open);
  }

  toggle(): boolean {
    this.setOpen(!this.open);
    return this.open;
  }

  update(telemetry: DiagnosticsTelemetry): void {
    if (!this.open) return;

    this.fpsRow.setValue(`${telemetry.fps.toFixed(1)} FPS`);
    this.fpsRow.setLevel(telemetry.fps < 40 ? 'warn' : 'normal');

    this.frameTimeRow.setValue(`${telemetry.frameTimeMs.toFixed(1)} ms`);
    this.frameTimeRow.setLevel(telemetry.frameTimeMs > 25 ? 'warn' : 'normal');

    this.gestureLatencyRow.setValue(
      telemetry.gestureLatencyMs > 0 ? `${telemetry.gestureLatencyMs.toFixed(1)} ms` : '— (STANDBY)'
    );

    this.inferenceFpsRow.setValue(
      telemetry.inferenceFps > 0 ? `${telemetry.inferenceFps} Hz` : '0 Hz'
    );

    this.ttftRow.setValue(
      telemetry.ttftMs > 0 ? `${telemetry.ttftMs} ms` : '— (NO RECENT CALL)'
    );

    this.aiTotalRow.setValue(
      telemetry.aiResponseMs > 0 ? `${telemetry.aiResponseMs} ms` : '—'
    );

    this.aiStateRow.setValue(telemetry.aiState);
    this.aiStateRow.setLevel(telemetry.aiState === 'ERROR' ? 'bad' : telemetry.aiState === 'THINKING' ? 'warn' : 'normal');

    this.cameraRow.setValue(telemetry.cameraState);
    this.micRow.setValue(telemetry.micState);

    this.backendRow.setValue(telemetry.backendConnected ? 'ONLINE (FASTAPI 2.0)' : 'OFFLINE / UNREACHABLE');
    this.backendRow.setLevel(telemetry.backendConnected ? 'normal' : 'bad');

    this.qualityRow.setValue(telemetry.qualityTier);
    this.particlesRow.setValue(telemetry.particles.toLocaleString());
    this.drawCallsRow.setValue(String(telemetry.drawCalls));

    this.errorsRow.setValue(String(telemetry.errorsCount));
    this.errorsRow.setLevel(telemetry.errorsCount > 0 ? 'bad' : 'normal');
  }
}
