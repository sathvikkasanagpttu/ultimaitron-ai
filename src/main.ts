import * as THREE from 'three';
import {
  AUDIO,
  IDENTITY,
  MAPPING,
  ORB,
  QUALITY,
  STORAGE_KEYS,
  type QualityTier,
} from './config';
import { detectCapabilities } from './utils/DeviceCapabilities';
import { PerformanceManager } from './utils/PerformanceManager';
import { clamp01 } from './utils/MathUtils';
import { attachContextHandlers, createRenderer, WebGLUnavailableError } from './scene/RendererFactory';
import { SceneManager } from './scene/SceneManager';
import { PostProcessing } from './scene/PostProcessing';
import { NeuralOrb } from './orb/NeuralOrb';
import { ORB_MODES, ORB_MODE_LABEL, type OrbMode } from './orb/OrbModes';
import { HolographicBeam } from './effects/HolographicBeam';
import { HandTrail } from './effects/HandTrail';
import { ParticleBurst } from './effects/ParticleBurst';
import { Reticle } from './effects/Reticle';
import { ScanArc } from './effects/ScanArc';
import { OrbitalNodes } from './effects/OrbitalNodes';
import { HandSkeleton } from './effects/HandSkeleton';
import { HandInteraction } from './interaction/HandInteraction';
import { GestureStateMachine, SYSTEM_GESTURE_LABEL } from './interaction/GestureStateMachine';
import { MouseFallback } from './interaction/MouseFallback';
import { InputAdapter } from './interaction/InputAdapter';
import { HandTracker } from './vision/HandTracker';
import { GESTURE_LABEL } from './vision/GestureEngine';
import { CAMERA_STATE_LABEL, type CameraState } from './vision/types';
import { HUD } from './ui/HUD';
import { BootSequence } from './ui/BootSequence';
import { AudioEngine } from './audio/AudioEngine';

import { AIStateManager } from './ai/AIStateManager';
import { AIClient } from './ai/AIClient';
import { VoiceAssistant } from './ai/VoiceAssistant';
import { CommandExecutor } from './ai/CommandExecutor';
import { ChatPanel } from './ui/ChatPanel';
import { DiagnosticsPanel } from './ui/DiagnosticsPanel';
import { KnowledgePanel } from './ui/KnowledgePanel';

const WARM_MODE_SECONDS = 2.4;

class Application {
  private readonly canvas: HTMLCanvasElement;
  private readonly ui: HTMLElement;

  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene: SceneManager;
  private readonly post: PostProcessing;
  private readonly orb: NeuralOrb;
  private readonly performance: PerformanceManager;

  private readonly beams: HolographicBeam[];
  private readonly trails: HandTrail[];
  private readonly burst: ParticleBurst;
  private readonly reticle = new Reticle();
  private readonly scanArc = new ScanArc();
  private readonly nodes = new OrbitalNodes();
  private readonly skeleton = new HandSkeleton();

  private readonly tracker = new HandTracker();
  private readonly gestures = new GestureStateMachine();
  private readonly mouse: MouseFallback;
  private readonly adapter = new InputAdapter();
  private readonly interaction: HandInteraction;

  private readonly hud: HUD;
  private readonly boot: BootSequence;
  private readonly audio = new AudioEngine();

  // Modern AI Companion Infrastructure
  private readonly aiState = new AIStateManager();
  private readonly aiClient = new AIClient();
  private readonly voice = new VoiceAssistant();
  private readonly executor: CommandExecutor;

  private readonly chatPanel: ChatPanel;
  private readonly diagnosticsPanel: DiagnosticsPanel;
  private readonly knowledgePanel: KnowledgePanel;

  private backendConnected = false;
  private errorsCount = 0;

  private lastFrameTime = 0;
  private pixelRatio: number;
  private tier: QualityTier;
  private elapsed = 0;
  private frameHandle = 0;
  private running = true;
  private contextLost = false;
  private warmTimer = 0;
  private cameraBusy = false;
  private reducedMotion: boolean;
  private highContrast: boolean;
  private lastCameraState: CameraState = 'OFF';

  constructor(canvas: HTMLCanvasElement, ui: HTMLElement) {
    this.canvas = canvas;
    this.ui = ui;

    const capabilities = detectCapabilities();
    this.tier = capabilities.suggestedTier;
    this.reducedMotion =
      readFlag(STORAGE_KEYS.reducedMotion) ?? capabilities.prefersReducedMotion;
    this.highContrast = readFlag(STORAGE_KEYS.highContrast) ?? false;
    const initialAudio = readFlag(STORAGE_KEYS.audio) ?? AUDIO.enabledByDefault;
    this.audio.init(initialAudio);

    const handle = createRenderer(canvas, this.tier);
    this.renderer = handle.renderer;
    this.pixelRatio = handle.pixelRatio;

    this.scene = new SceneManager(window.innerWidth / window.innerHeight, this.pixelRatio);
    this.orb = new NeuralOrb(this.tier, this.pixelRatio);
    this.scene.add(this.orb.group);

    this.beams = [new HolographicBeam(0), new HolographicBeam(1)];
    this.trails = [new HandTrail(this.pixelRatio), new HandTrail(this.pixelRatio)];
    this.burst = new ParticleBurst(this.pixelRatio);

    for (const beam of this.beams) this.scene.add(beam.mesh);
    for (const trail of this.trails) this.scene.add(trail.points);
    this.scene.add(this.burst.group);
    this.scene.add(this.reticle.group);
    this.scene.add(this.scanArc.mesh);
    this.scene.add(this.nodes.group);
    this.scene.add(this.skeleton.group);

    this.post = new PostProcessing(
      this.renderer,
      this.scene.scene,
      this.scene.rig.camera,
      this.tier,
      window.innerWidth,
      window.innerHeight,
    );

    this.performance = new PerformanceManager(this.tier, true);

    this.mouse = new MouseFallback(canvas);
    this.mouse.attach();
    this.mouse.setEnabled(true);

    this.interaction = new HandInteraction(this.orb, this.scene.rig, {
      beams: this.beams,
      trails: this.trails,
      reticle: this.reticle,
      scanArc: this.scanArc,
      nodes: this.nodes,
      skeleton: this.skeleton,
    });

    this.hud = new HUD(ui);
    this.boot = new BootSequence(ui, this.reducedMotion);

    // AI Companion UI panels
    this.chatPanel = new ChatPanel(ui, {
      onSendMessage: (text) => void this.handleUserMessage(text),
      onToggleVoice: () => void this.toggleVoice(),
      onClearChat: () => {
        this.aiClient.clear();
        this.hud.gestures.toast('CONVERSATION CLEARED');
      },
    });

    this.diagnosticsPanel = new DiagnosticsPanel(ui);
    this.knowledgePanel = new KnowledgePanel(ui);

    // Central safe Command Registry executor
    this.executor = new CommandExecutor({
      setMode: (mode) => {
        this.orb.setMode(mode);
        this.changeMode(mode);
      },
      firePulse: (_intensity) => {
        this.orb.firePulse();
        this.audio.play('power');
      },
      triggerBurst: (warm) => {
        this.orb.triggerBurst();
        this.burst.fire(
          this.orb.group.position,
          ORB.radius * 0.4 * this.orb.currentScale,
          Math.round(320 * (this.tier === 'LOW' ? 0.5 : 1)),
          4.4,
          warm ?? 0,
        );
        this.audio.play('power');
      },
      setScale: (scale) => {
        this.orb.setScaleTarget(scale);
      },
      setWarmth: (warmth) => {
        this.orb.setWarmTarget(warmth);
      },
      resetSystem: () => {
        this.orb.resetOrientation();
        this.interaction.reset();
        this.scene.rig.setDistance(6.2);
        this.changeMode('NEURAL');
      },
      openPanel: (panel) => {
        if (panel === 'chat') this.chatPanel.setOpen(true);
        else if (panel === 'diagnostics') this.diagnosticsPanel.setOpen(true);
        else if (panel === 'knowledge') this.knowledgePanel.setOpen(true);
      },
    });

    this.setupAISystem();
    this.applyAccessibility();
    this.registerControls(initialAudio);
    this.registerEvents();

    if (initialAudio) {
      const unlock = () => {
        if (this.audio.isEnabled) {
          void this.audio.setEnabled(true);
        }
        window.removeEventListener('pointerdown', unlock);
        window.removeEventListener('keydown', unlock);
      };
      window.addEventListener('pointerdown', unlock, { passive: true });
      window.addEventListener('keydown', unlock, { passive: true });
    }

    this.boot.onFormationStart = () => {
      if (this.reducedMotion) this.orb.completeFormation();
      else this.orb.startFormation();
      this.audio.play('boot');
    };
    this.boot.onComplete = () => {
      this.nodes.setVisible(true);
      this.hud.gestures.toast('READY FOR INTERACTION');
      this.orb.firePulse();
    };

    this.hud.gestures.setMode(ORB_MODE_LABEL.NEURAL, 0, ORB_MODES.length, false);
    this.hud.status.setStatus('ONLINE');

    this.resize();
    this.lastFrameTime = performance.now();
    this.loop();
  }

  /* ------------------------------------------------------------------ */
  /* AI & voice initialization                                          */
  /* ------------------------------------------------------------------ */

  private setupAISystem(): void {
    void this.checkBackend();

    // AI state machine drives the holographic core shader parameters
    this.aiState.events.on('change', ({ to }) => {
      const profile = this.aiState.profile;
      this.orb.setEnergyTarget(profile.energyTarget);
      this.orb.setWarmTarget(profile.warmthTarget);
      this.post.setAlert(profile.alert);
      this.orb.firePulse();

      this.chatPanel.setState(to, to === 'ERROR' ? 'warn' : to === 'IDLE' ? 'normal' : 'active');
      this.hud.gestures.toast(`AI CORE · ${to}`);
    });

    // AI Client SSE event handling
    this.aiClient.events.on('token', (token) => {
      this.chatPanel.appendToken(token);
    });

    this.aiClient.events.on('toolCall', (call) => {
      this.chatPanel.appendToolCall(call);
      const res = this.executor.execute(call);
      this.hud.gestures.toast(res.message, res.success ? 'normal' : 'warn');
    });

    this.aiClient.events.on('citation', (citations) => {
      this.chatPanel.appendCitations(citations);
    });

    this.aiClient.events.on('done', ({ fullText }) => {
      this.chatPanel.endStream();
      if (!this.voice.isMuted) {
        this.voice.speak(fullText);
      } else {
        this.aiState.setState('IDLE');
      }
    });

    this.aiClient.events.on('error', (err) => {
      this.chatPanel.appendError(err);
      this.aiState.setState('ERROR');
      this.errorsCount++;
      this.hud.gestures.toast(err.slice(0, 50), 'warn');
    });

    // Voice recognition & synthesis events
    this.voice.events.on('listeningStart', () => {
      this.aiState.setState('LISTENING');
      this.chatPanel.setMicActive(true);
      this.hud.status.setControlState('voice', 'LISTENING', true);
      this.hud.gestures.toast('LISTENING FOR SPEECH...');
    });

    this.voice.events.on('listeningEnd', () => {
      if (this.aiState.state === 'LISTENING') {
        this.aiState.setState('IDLE');
      }
      this.chatPanel.setMicActive(false);
      this.hud.status.setControlState('voice', 'OFF', false);
    });

    this.voice.events.on('transcript', ({ text, isFinal }) => {
      if (isFinal) {
        this.chatPanel.setInputValue('');
        void this.handleUserMessage(text);
      } else {
        this.chatPanel.setInputValue(text);
      }
    });

    this.voice.events.on('speakingStart', () => {
      this.aiState.setState('SPEAKING');
    });

    this.voice.events.on('speakingEnd', () => {
      if (this.aiState.state === 'SPEAKING') {
        this.aiState.setState('IDLE');
      }
    });

    this.voice.events.on('error', (err) => {
      this.hud.gestures.toast(err, 'warn');
    });
  }

  private async checkBackend(): Promise<void> {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        this.backendConnected = data.status === 'ok';
      } else {
        this.backendConnected = false;
      }
    } catch {
      this.backendConnected = false;
    }
  }

  private async handleUserMessage(text: string): Promise<void> {
    if (!text.trim()) return;
    this.chatPanel.setOpen(true);
    this.aiState.setState('THINKING');
    this.chatPanel.beginStream();
    await this.aiClient.sendMessage(text);
  }

  private toggleVoice(): void {
    const active = this.voice.toggleListening();
    this.hud.status.setControlState('voice', active ? 'LISTENING' : 'OFF', active);
  }

  /* ------------------------------------------------------------------ */
  /* wiring                                                              */
  /* ------------------------------------------------------------------ */

  private registerControls(initialAudio: boolean): void {
    this.hud.status.addControl(
      { id: 'chat', label: 'AI CHAT', initialState: 'OFF' },
      () => {
        const open = this.chatPanel.toggle();
        this.hud.status.setControlState('chat', open ? 'OPEN' : 'OFF', open);
      },
    );
    this.hud.status.addControl(
      { id: 'voice', label: 'VOICE STT/TTS', initialState: 'OFF' },
      () => this.toggleVoice(),
    );
    this.hud.status.addControl(
      { id: 'diagnostics', label: 'DIAGNOSTICS', initialState: 'OFF' },
      () => {
        const open = this.diagnosticsPanel.toggle();
        this.hud.status.setControlState('diagnostics', open ? 'ON' : 'OFF', open);
      },
    );
    this.hud.status.addControl(
      { id: 'vault', label: 'KNOWLEDGE VAULT', initialState: 'OFF' },
      () => {
        const open = this.knowledgePanel.toggle();
        this.hud.status.setControlState('vault', open ? 'OPEN' : 'OFF', open);
      },
    );
    this.hud.status.addControl(
      { id: 'camera', label: 'CAMERA', initialState: 'OFF' },
      () => void this.toggleCamera(),
    );
    this.hud.status.addControl(
      { id: 'sound', label: 'SOUND', initialState: initialAudio ? 'ON' : 'OFF', on: initialAudio },
      () => void this.toggleAudio(),
    );
    this.hud.status.addControl({ id: 'debug', label: 'DEBUG', initialState: 'OFF' }, () =>
      this.toggleDebug(),
    );
    this.hud.status.addControl({ id: 'hud', label: 'INTERFACE', initialState: 'ON', on: true }, () => {
      const visible = this.hud.toggleVisible();
      this.hud.status.setControlState('hud', visible ? 'ON' : 'OFF', visible);
    });
    this.hud.status.addControl(
      {
        id: 'motion',
        label: 'REDUCED MOTION',
        initialState: this.reducedMotion ? 'ON' : 'OFF',
        on: this.reducedMotion,
      },
      () => this.setReducedMotion(!this.reducedMotion),
    );
    this.hud.status.addControl(
      {
        id: 'contrast',
        label: 'HIGH CONTRAST',
        initialState: this.highContrast ? 'ON' : 'OFF',
        on: this.highContrast,
      },
      () => this.setHighContrast(!this.highContrast),
    );
  }

  private registerEvents(): void {
    window.addEventListener('resize', this.resize);
    window.addEventListener('keydown', this.onKeyDown);
    document.addEventListener('visibilitychange', this.onVisibilityChange);

    attachContextHandlers(
      this.canvas,
      () => {
        this.contextLost = true;
        this.hud.gestures.toast('GRAPHICS CONTEXT LOST', 'warn');
        this.hud.status.setStatus('RECOVERING', 'bad');
        this.post.setAlert(true);
      },
      () => {
        this.contextLost = false;
        this.hud.gestures.toast('GRAPHICS CONTEXT RESTORED');
        this.hud.status.setStatus('ONLINE');
        this.post.setAlert(false);
        this.resize();
      },
    );

    this.performance.onQualityChange((tier, reason) => {
      this.applyQuality(tier);
      this.hud.gestures.toast(
        reason === 'downgrade' ? `QUALITY REDUCED — ${tier}` : `QUALITY RESTORED — ${tier}`,
        reason === 'downgrade' ? 'warn' : 'normal',
      );
    });

    /* --- vision ------------------------------------------------------- */
    this.tracker.events.on('state', (state) => this.onCameraState(state));
    this.tracker.events.on('inference', () => this.performance.markInference(performance.now()));
    this.tracker.events.on('error', (error) => {
      this.hud.status.showNotice({
        title: error.kind === 'denied' ? 'CAMERA ACCESS REQUIRED' : 'CAMERA UNAVAILABLE',
        body:
          error.kind === 'denied'
            ? 'Gesture control needs the webcam. Allow camera access in your browser, then press CAMERA again. Video never leaves this device — every frame is processed locally.'
            : `${error.message} You can keep using the interface with the mouse: move to steer, hold the left button to pinch, scroll to zoom.`,
        actions: [
          { label: 'CONTINUE WITHOUT CAMERA', onClick: () => this.hud.status.hideNotice() },
          { label: 'RETRY', onClick: () => void this.retryCamera() },
        ],
      });
      this.hud.status.setControlState('camera', 'DENIED', false);
      this.mouse.setEnabled(true);
    });

    /* --- gestures ----------------------------------------------------- */
    this.gestures.events.on('change', ({ to }) => {
      this.hud.gestures.notifyActivity();
      if (to === 'IDLE') return;
      this.hud.gestures.toast(SYSTEM_GESTURE_LABEL[to]);
      if (to === 'HAND_DETECTED') this.audio.play('handDetected');
      if (to === 'OPEN_PALM') this.orb.firePulse();
      if (to === 'THUMBS_UP') {
        this.audio.play('select');
        this.orb.firePulse();
        this.hud.gestures.toast('ACTION CONFIRMED', 'normal');
      }
    });

    this.gestures.events.on('pinchStart', () => {
      this.audio.play('pinch');
      this.hud.gestures.notifyActivity();
    });
    this.gestures.events.on('pinchEnd', () => this.audio.play('release'));

    this.gestures.events.on('fistStart', () => {
      this.orb.setContract(1);
      this.orb.setWarmTarget(1);
      this.warmTimer = WARM_MODE_SECONDS;
    });

    this.gestures.events.on('fistRelease', () => {
      this.orb.setContract(0);
      this.orb.triggerBurst();
      this.burst.fire(
        this.orb.group.position,
        ORB.radius * 0.4 * this.orb.currentScale,
        Math.round(320 * (this.tier === 'LOW' ? 0.5 : 1)),
        4.4,
        1,
      );
      this.audio.play('power');
      this.hud.gestures.toast('POWER SURGE');
      this.warmTimer = WARM_MODE_SECONDS;
    });

    this.gestures.events.on('swipe', ({ direction }) => {
      // The view is mirrored, so an on-screen right swipe is a decreasing x.
      const applied = MAPPING.mirrorX ? -direction : direction;
      this.changeMode(this.orb.cycleMode(applied));
    });

    this.nodes.onSelect = (node) => {
      this.audio.play('select');
      this.hud.gestures.toast(`${node.label} — ${node.detail}`);
    };

    this.boot.onStep = (step) => {
      if (step.final) this.audio.play('mode');
    };

    // A click anywhere skips the boot theatre.
    this.canvas.addEventListener('pointerdown', () => this.boot.skip(), { once: true });
  }

  /* ------------------------------------------------------------------ */
  /* controls                                                            */
  /* ------------------------------------------------------------------ */

  private async toggleCamera(): Promise<void> {
    if (this.cameraBusy) return;
    this.cameraBusy = true;
    this.hud.status.hideNotice();

    try {
      if (this.tracker.isEnabled) {
        this.tracker.disable();
        this.mouse.setEnabled(true);
        this.hud.debug.detachVideo();
        this.hud.status.setControlState('camera', 'OFF', false);
        this.interaction.reset();
        return;
      }

      this.hud.status.setControlState('camera', 'STARTING…', false);
      const started = await this.tracker.enable();
      if (started) {
        this.mouse.setEnabled(false);
        this.hud.debug.attachVideo(this.tracker.feed.video);
        this.hud.status.setControlState('camera', 'ON', true);
        this.hud.gestures.toast('OPTICAL INPUT ONLINE');
        this.hud.gestures.notifyActivity();
      } else {
        this.hud.status.setControlState('camera', 'OFF', false);
        this.mouse.setEnabled(true);
      }
    } finally {
      this.cameraBusy = false;
    }
  }

  private async retryCamera(): Promise<void> {
    this.hud.status.hideNotice();
    await this.toggleCamera();
  }

  private async toggleAudio(): Promise<void> {
    const enabled = await this.audio.setEnabled(!this.audio.isEnabled);
    this.hud.status.setControlState('sound', enabled ? 'ON' : 'OFF', enabled);
    window.localStorage?.setItem(STORAGE_KEYS.audio, String(enabled));
    if (enabled) this.audio.play('select');
  }

  private toggleDebug(): void {
    const open = this.hud.debug.toggle();
    this.skeleton.setVisible(open);
    this.hud.status.setControlState('debug', open ? 'ON' : 'OFF', open);
    if (open && this.tracker.isEnabled) this.hud.debug.attachVideo(this.tracker.feed.video);
  }

  private setReducedMotion(enabled: boolean): void {
    this.reducedMotion = enabled;
    this.applyAccessibility();
    this.hud.status.setControlState('motion', enabled ? 'ON' : 'OFF', enabled);
    window.localStorage?.setItem(STORAGE_KEYS.reducedMotion, String(enabled));
  }

  private setHighContrast(enabled: boolean): void {
    this.highContrast = enabled;
    this.applyAccessibility();
    this.hud.status.setControlState('contrast', enabled ? 'ON' : 'OFF', enabled);
    window.localStorage?.setItem(STORAGE_KEYS.highContrast, String(enabled));
  }

  private applyAccessibility(): void {
    document.body.classList.toggle('reduced-motion', this.reducedMotion);
    document.body.classList.toggle('high-contrast', this.highContrast);
    this.orb.setReducedMotion(this.reducedMotion);
    this.scene.rig.setReducedMotion(this.reducedMotion);
    this.post.setHighContrast(this.highContrast);
  }

  private changeMode(mode: OrbMode): void {
    this.hud.gestures.setMode(ORB_MODE_LABEL[mode], ORB_MODES.indexOf(mode), ORB_MODES.length);
    this.hud.gestures.toast(`MODE — ${ORB_MODE_LABEL[mode]}`);
    this.audio.play('mode');
    this.orb.firePulse();
  }

  private applyQuality(tier: QualityTier): void {
    this.tier = tier;
    this.orb.setQuality(tier);
    this.post.setQuality(tier);
    this.pixelRatio = Math.min(window.devicePixelRatio || 1, QUALITY[tier].pixelRatioCap);
    this.renderer.setPixelRatio(this.pixelRatio);
    this.orb.setPixelRatio(this.pixelRatio);
    this.scene.setPixelRatio(this.pixelRatio);
    this.burst.setPixelRatio(this.pixelRatio);
    for (const trail of this.trails) trail.setPixelRatio(this.pixelRatio);
    this.resize();
  }

  private onCameraState(state: CameraState): void {
    this.lastCameraState = state;
    if (state === 'TRACKING_LOST' || state === 'NO_HAND') {
      this.post.setAlert(state === 'TRACKING_LOST');
      if (state === 'TRACKING_LOST') this.audio.play('handLost');
    } else {
      this.post.setAlert(false);
    }
  }

  private readonly onKeyDown = (event: KeyboardEvent): void => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const key = event.key.toLowerCase();

    if (key >= '1' && key <= '5') {
      const mode = ORB_MODES[Number(key) - 1];
      if (mode) {
        this.orb.setMode(mode);
        this.changeMode(mode);
      }
      return;
    }

    switch (key) {
      case 'g':
        void this.toggleCamera();
        break;
      case 't':
        this.chatPanel.toggle();
        break;
      case 'v':
        this.toggleVoice();
        break;
      case 'x':
        this.diagnosticsPanel.toggle();
        break;
      case 'k':
        this.knowledgePanel.toggle();
        break;
      case ' ':
        event.preventDefault();
        this.orb.firePulse();
        this.orb.triggerBurst();
        this.burst.fire(this.orb.group.position, ORB.radius * 0.4 * this.orb.currentScale, 260, 4.0, 0);
        this.audio.play('power');
        this.hud.gestures.toast('POWER PULSE');
        break;
      case 'r':
        this.orb.resetOrientation();
        this.interaction.reset();
        this.scene.rig.setDistance(6.2);
        this.hud.gestures.setMode(ORB_MODE_LABEL.NEURAL, 0, ORB_MODES.length);
        this.hud.gestures.toast('SYSTEM RESET');
        break;
      case 'd':
        this.toggleDebug();
        break;
      case 'h': {
        const visible = this.hud.toggleVisible();
        this.hud.status.setControlState('hud', visible ? 'ON' : 'OFF', visible);
        break;
      }
      case 'escape':
        this.hud.status.hideNotice();
        this.chatPanel.setOpen(false);
        this.diagnosticsPanel.setOpen(false);
        this.knowledgePanel.setOpen(false);
        break;
      default:
        break;
    }
  };

  private readonly onVisibilityChange = (): void => {
    this.running = !document.hidden;
    if (this.running) {
      this.lastFrameTime = performance.now();
      this.loop();
    }
  };

  private readonly resize = (): void => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    this.renderer.setSize(width, height, false);
    this.scene.resize(width, height);
    this.post.setSize(width, height);
  };

  /* ------------------------------------------------------------------ */
  /* frame loop                                                          */
  /* ------------------------------------------------------------------ */

  private readonly loop = (): void => {
    if (!this.running) return;
    this.frameHandle = requestAnimationFrame(this.loop);

    const now = performance.now();
    this.renderer.info.reset();

    const rawDt = (now - this.lastFrameTime) / 1000;
    const dt = Math.min(0.05, rawDt);
    this.lastFrameTime = now;
    this.elapsed += dt;

    this.performance.update(rawDt);
    this.boot.update(dt);
    this.aiState.update(dt);

    // Phonetic voice cadence drives orb surface ripple
    const voiceAmp = this.voice.update(dt);
    if (this.voice.isSpeaking && voiceAmp > 0) {
      this.orb.setEnergyTarget(Math.min(1.0, 0.72 + voiceAmp * 0.28));
    }

    /* --- input --------------------------------------------------------- */
    this.tracker.update(now);

    const usingCamera = this.tracker.isEnabled;
    const hands = usingCamera
      ? this.adapter.fromTracked(this.tracker.hands)
      : this.adapter.fromPointer(this.mouse.update(dt));

    if (!usingCamera) {
      const zoom = this.mouse.consumeZoom();
      if (zoom !== 0) {
        this.orb.nudgeScale(zoom * 1.2);
        this.scene.rig.nudgeDistance(-zoom * 2.2);
      }
    }

    const systemGesture = this.gestures.update(hands, dt);

    /* --- world --------------------------------------------------------- */
    const camera = this.scene.rig.camera;
    this.interaction.update(dt, this.elapsed, hands, systemGesture, camera.quaternion);

    if (this.warmTimer > 0) {
      this.warmTimer -= dt;
      if (this.warmTimer <= 0) this.orb.setWarmTarget(0);
    }

    this.orb.update(dt, this.elapsed, camera);
    this.burst.update(dt, this.elapsed, camera.quaternion);
    this.scene.update(dt, this.elapsed, this.orb.activation, this.orb.warmth);
    this.post.update(dt, this.elapsed, this.orb.activation);

    /* --- render -------------------------------------------------------- */
    if (!this.contextLost) {
      this.post.render();
    }

    this.updateHud(dt, systemGesture, usingCamera);
  };

  private updateHud(dt: number, systemGesture: string, usingCamera: boolean): void {
    const hands = this.adapter.current;
    const frameTime = this.performance.frameTimeMs;
    const gpuEstimate = clamp01(frameTime / 16.67);

    const cameraLabel = usingCamera
      ? CAMERA_STATE_LABEL[this.lastCameraState]
      : this.tracker.state === 'DENIED'
        ? 'POINTER (CAMERA DENIED)'
        : 'POINTER CONTROL';

    const level =
      this.lastCameraState === 'TRACKING_LOST' || this.lastCameraState === 'ERROR'
        ? 'bad'
        : usingCamera && this.lastCameraState !== 'TRACKING'
          ? 'warn'
          : 'ok';

    const primaryGesture = hands[0]?.active
      ? GESTURE_LABEL[hands[0].gesture]
      : hands[1]?.active
        ? GESTURE_LABEL[hands[1].gesture]
        : 'STANDBY';

    const activation = this.orb.activation;
    this.hud.update(dt, {
      cameraLabel,
      cameraLevel: level,
      hands: [
        {
          present: hands[0].active,
          label: hands[0].label,
          confidence: hands[0].confidence,
          gesture: GESTURE_LABEL[hands[0].gesture],
        },
        {
          present: hands[1].active,
          label: hands[1].label,
          confidence: hands[1].confidence,
          gesture: GESTURE_LABEL[hands[1].gesture],
        },
      ],
      gestureLabel: primaryGesture,
      handCount: (hands[0].active ? 1 : 0) + (hands[1].active ? 1 : 0),
      power: activation,
      neural: clamp01(activation * 0.72 + 0.2 + Math.sin(this.elapsed * 1.3) * 0.05),
      gpuEstimate,
      latencyMs: usingCamera ? this.tracker.lastInferenceMs : 0,
      fps: this.performance.fps,
      quality: this.tier,
      processing: usingCamera ? `VISION / ${this.tracker.visionMode.toUpperCase()}` : 'POINTER',
      energyLabel: activation > 0.85 ? 'SURGE' : activation > 0.5 ? 'ELEVATED' : 'NOMINAL',
    });

    // Update real diagnostics panel metrics (zero allocation, only updates open panel)
    this.diagnosticsPanel.update({
      fps: this.performance.fps,
      frameTimeMs: this.performance.frameTimeMs,
      gestureLatencyMs: usingCamera ? this.tracker.lastInferenceMs : 0,
      inferenceFps: this.performance.inferenceFps,
      ttftMs: this.aiClient.ttftMs,
      aiResponseMs: this.aiClient.totalMs,
      aiState: this.aiState.state,
      cameraState: cameraLabel,
      micState: this.voice.isListening ? 'LISTENING' : this.voice.isSpeaking ? 'SPEAKING' : 'STANDBY',
      backendConnected: this.backendConnected,
      qualityTier: this.tier,
      particles: this.orb.stats.particles,
      connections: this.orb.stats.connections,
      drawCalls: this.renderer.info.render.calls,
      errorsCount: this.errorsCount,
    });

    if (this.hud.debug.isOpen) {
      const info = this.renderer.info;
      this.hud.debug.update({
        fps: this.performance.fps,
        inferenceFps: this.performance.inferenceFps,
        inferenceMs: this.tracker.lastInferenceMs,
        visionMode: usingCamera ? this.tracker.visionMode : 'pointer',
        delegate: usingCamera ? this.tracker.delegate : '—',
        cameraState: this.lastCameraState,
        gestureState: systemGesture,
        confidence: Math.max(hands[0].confidence, hands[1].confidence),
        handWorld: formatVector(this.interaction.worldPosition(0)),
        pinchDistance: hands[0].active ? hands[0].pinchDistance : hands[1].pinchDistance,
        particles: this.orb.stats.particles,
        connections: this.orb.stats.connections,
        motes: this.orb.stats.motes,
        quality: this.tier,
        drawCalls: info.render.calls,
        triangles: info.render.triangles,
        programs: info.programs?.length ?? 0,
        pixelRatio: this.pixelRatio,
      });
      this.hud.debug.updateAssistant({
        assistantState: this.aiState.state,
        backend: this.backendConnected ? 'ONLINE' : 'OFFLINE',
        aiResponseMs: this.aiClient.totalMs > 0 ? `${this.aiClient.totalMs.toFixed(0)}ms` : '—',
        firstTokenMs: this.aiClient.ttftMs > 0 ? `${this.aiClient.ttftMs.toFixed(0)}ms` : '—',
      });
      this.hud.debug.drawLandmarks([hands[0].landmarks, hands[1].landmarks]);
    }
  }

  dispose(): void {
    this.running = false;
    cancelAnimationFrame(this.frameHandle);
    window.removeEventListener('resize', this.resize);
    window.removeEventListener('keydown', this.onKeyDown);
    document.removeEventListener('visibilitychange', this.onVisibilityChange);

    this.tracker.dispose();
    this.voice.dispose();
    this.aiClient.abort();
    this.mouse.dispose();
    this.orb.dispose();
    this.scene.dispose();
    this.post.dispose();
    this.burst.dispose();
    this.reticle.dispose();
    this.scanArc.dispose();
    this.nodes.dispose();
    this.skeleton.dispose();
    for (const beam of this.beams) beam.dispose();
    for (const trail of this.trails) trail.dispose();
    this.audio.dispose();
    this.renderer.dispose();
    this.ui.replaceChildren();
  }
}

/* -------------------------------------------------------------------- */
/* bootstrap                                                             */
/* -------------------------------------------------------------------- */

function formatVector(vector: THREE.Vector3): string {
  return `${vector.x.toFixed(2)}, ${vector.y.toFixed(2)}, ${vector.z.toFixed(2)}`;
}

function readFlag(key: string): boolean | null {
  try {
    const raw = window.localStorage?.getItem(key);
    return raw === null || raw === undefined ? null : raw === 'true';
  } catch {
    return null;
  }
}

function showFatal(title: string, body: string): void {
  const overlay = document.createElement('div');
  overlay.className = 'fatal';
  const heading = document.createElement('h1');
  heading.textContent = title;
  const paragraph = document.createElement('p');
  paragraph.textContent = body;
  overlay.append(heading, paragraph);
  document.body.append(overlay);
}

function start(): void {
  const canvas = document.getElementById('scene');
  const ui = document.getElementById('ui');

  if (!(canvas instanceof HTMLCanvasElement) || !ui) {
    showFatal('INTERFACE FAILED TO MOUNT', 'The application shell could not be found in the page.');
    return;
  }

  const capabilities = detectCapabilities();
  if (!capabilities.webgl) {
    showFatal(
      'WEBGL ENGINE UNAVAILABLE',
      'Please use a modern browser with hardware acceleration enabled.',
    );
    return;
  }

  try {
    const app = new Application(canvas, ui);
    const win = window as unknown as { ultimaitron?: unknown; nexus?: unknown };
    win.ultimaitron = app;
    win.nexus = app;
    window.addEventListener('beforeunload', () => app.dispose());
  } catch (error) {
    console.error(`[${IDENTITY.name}] failed to start:`, error);
    if (error instanceof WebGLUnavailableError) {
      showFatal(
        'WEBGL ENGINE UNAVAILABLE',
        'Please use a modern browser with hardware acceleration enabled.',
      );
    } else {
      showFatal(
        'CORE INITIALISATION FAILED',
        error instanceof Error ? error.message : 'An unknown error prevented start-up.',
      );
    }
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', start, { once: true });
} else {
  start();
}
