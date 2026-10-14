# Ultimaitron AI — Neural Intelligence Core

[![Three.js](https://img.shields.io/badge/Three.js-r185-black?style=for-the-badge&logo=three.js)](https://threejs.org/)
[![WebGL 2.0](https://img.shields.io/badge/WebGL-2.0-red?style=for-the-badge&logo=webgl)](https://www.khronos.org/webgl/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks_Vision-blue?style=for-the-badge&logo=google)](https://developers.google.com/mediapipe)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue?style=for-the-badge&logo=typescript)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-5.4-purple?style=for-the-badge&logo=vite)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Ultimaitron AI** is a state-of-the-art, gesture-controlled holographic neural companion interface. Rendered in real time using **Three.js**, custom **GLSL** shaders, and **MediaPipe Hand Landmarker**, it visualizes a sentient artificial intelligence core driven by GPU-computed particle mechanics, synaptic impulse networks, layered volumetric energy fields, and spatial holographic user interactions.

---

## Visual Architecture Overview

```
                          ┌───────────────────────────┐
                          │   WEBCAM / OPTICAL INPUT  │
                          └─────────────┬─────────────┘
                                        │ 640x480 @ 30 FPS
                                        ▼
                          ┌───────────────────────────┐
                          │   VISION WEB WORKER       │
                          │   MediaPipe HandLandmarker│
                          └─────────────┬─────────────┘
                                        │ 21 3D Landmarks (Normalized)
                                        ▼
                          ┌───────────────────────────┐
                          │   FILTER & GESTURE ENGINE │
                          │   • 1€ Filter (Jitter-free)│
                          │   • Hysteresis Latches    │
                          │   • Debounced State Mach. │
                          └─────────────┬─────────────┘
                                        │ InputHand[2] + SystemGesture
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                       HAND INTERACTION SYSTEM                          │
    │   • Hand -> Holographic World Projection (X/Y/Depth Scaling)          │
    │   • Holographic Beams, Comet Trails, Reticle & Scan Arc Targeting      │
    │   • Dual-Hand Spatial Scaling & Continuous Rotation Physics           │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │ Transforms & Uniform Writes
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                             GPU RENDERING PIPELINE                                 │
│                                                                                    │
│   ┌─────────────────────┐   ┌─────────────────────┐   ┌────────────────────────┐   │
│   │   NEURAL PARTICLES  │   │   SYNAPSE NETWORK   │   │       ENERGY CORE      │   │
│   │  • 20k Nodes (HIGH) │   │  • 8k Lines (HIGH)  │   │  • Dark Occlusion Core │   │
│   │  • 5 Morph Targets  │   │  • Welded to Nodes  │   │  • Procedural Plasma   │   │
│   │  • Wavefront Pulses │   │  • Signal Impulses  │   │  • Containment Torus   │   │
│   │  • Noise Breathing  │   │  • Spatial Hash O(N)│   │  • Radial Billboard Ray│   │
│   └──────────┬──────────┘   └──────────┬──────────┘   └───────────┬────────────┘   │
│              │                         │                          │                │
│   ┌──────────┴──────────┐   ┌──────────┴──────────┐   ┌───────────┴────────────┐   │
│   │    ORBITAL RINGS    │   │     DATA MOTES      │   │    ENVIRONMENT HAZE    │   │
│   │  • 9 Dynamic Planes │   │  • 1.4k Free Drifters│  │  • Star Field Sphere   │   │
│   │  • Staggered Wobble │   │  • Orbital Physics  │   │  • Core Backing Fog    │   │
│   └──────────┬──────────┘   └──────────┬──────────┘   └───────────┬────────────┘   │
│              └─────────────────────────┼──────────────────────────┘                │
│                                        ▼                                           │
│                     THREE.JS SCENE GRAPH (SceneManager)                            │
│                                        │                                           │
│                                        ▼                                           │
│                     COMPOSER & POST-PROCESSING STACK                               │
│                     • RenderPass (Physical Geometry)                               │
│                     • UnrealBloomPass (Selective HDR Threshold)                    │
│                     • Composite Shader (Aberration, Vignette, Scanlines, Grain)    │
│                                        │                                           │
│                                        ▼                                           │
│                               DISPLAY CANVAS (60+ FPS)                             │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3D Animation & Real-Time VFX Engineering

Ultimaitron AI moves beyond standard particle canvas demos by adhering to high-end creative-technology principles: **zero CPU allocations per frame**, **complete vertex-shader animation**, **deterministic geometry synthesis**, and **physically grounded holographic shading**.

### 1. Unified Procedural Morphing & GPU Displacement
The neural core does not stream animated vertices from the CPU. All particles are stored in a single immutable `THREE.BufferGeometry` with multiple parallel base coordinates:
* `position`: Base resting coordinates for **NEURAL** configuration.
* `aReactor`: Concentric spoke discs with alternating tilts for **REACTOR CORE**.
* `aData`: High-density mathematical lattice sphere for **DATA SPHERE**.
* `aEnergy`: Turbulent curl-noise plasma dispersion for **ENERGY FIELD**.
* `aScanner`: Dual planar cross-section slicing slabs for **SCANNER ARRAY**.

In [common.glsl](file:///Users/kasanagotttusathvik/Downloads/ULTIMAITRON-UI-main/src/shaders/common.glsl), dynamic runtime morphing is computed on the GPU via continuous $n$-way barycentric interpolation:

$$\vec{P}_{\text{base}} = w_0 \vec{P}_{\text{neural}} + w_1 \vec{P}_{\text{reactor}} + w_2 \vec{P}_{\text{data}} + w_3 \vec{P}_{\text{energy}} + w_4 \vec{P}_{\text{scanner}}$$

$$\sum_{i=0}^{4} w_i = 1$$

Because the connection lines (`THREE.LineSegments`) share these exact vertex attributes and execute the identical displacement function, **the synapses remain mathematically welded to their parent neurons** throughout all deformations, contractions, and power bursts without any CPU graph maintenance.

### 2. Multi-Wavefront Energy Pulses
Computation activity is visualized as expanding spherical wavefront shockwaves radiating through the neural mesh. Up to 4 concurrent shockwaves are held in uniform vectors `uPulses[4]` ($\vec{O}_{xyz}, t_{\text{start}}$).

Every vertex evaluates its Euclidean distance to each wavefront:

$$A(p, t) = \sum_{i=0}^{3} \exp\left( - \left( \frac{\|\vec{p} - \vec{O}_i\| - (t - t_{0,i}) \cdot v_{\text{pulse}}}{w_{\text{pulse}}} \right)^2 \right) \cdot \exp\left( -(t - t_{0,i}) \cdot \lambda_{\text{decay}} \right)$$

This calculation creates radiant crests that surge outward across the topology when firing random idle synapses, user gestures, or AI companion responses.

### 3. Layered Holographic Energy Core
The center of the orb is an intricate 6-layer optical assembly:
1. **Dark Occlusion Sphere** ($r = 0.185$): Writes to the depth buffer to occlude back-facing particles and provide physical depth.
2. **Procedural Plasma Core** ($r = 0.225$): High-frequency noise displacement with dual-tone Fresnel rim lighting.
3. **Containment Ring**: Tilted torus rotating dynamically around the core axis.
4. **Radial Ray Fan**: 64-segment holographic aperture plane billboarded to face the camera rig.
5. **Translucent Outer Shell** ($r = 0.44$): Subtle inverted-Fresnel glass envelope.
6. **Volumetric Bloom Sprite**: Camera-facing radial gradient that breathes harmonically with core activation.

### 4. Concentric Orbital Rings & Staggered Machinery
Nine independent concentric rings orbit the core with deliberately non-uniform radii, inclinations, and phase offsets:
* **Asymmetric Kinematics**: Each ring possesses distinct tilt angles ($X/Y$), rotation velocity, and harmonic wobble frequencies ($A_1 \sin(\omega_1 t) + A_2 \cos(\omega_2 t)$).
* **Dual Hologram Shader**: Procedural tick dashes, segment division, animated sweeps, and edge feathering rendered on two-sided additive ring geometry.
* **Dynamic Acceleration**: When hand presence is detected, rings accelerate smoothly from idle drift ($0.35\,\text{rad/s}$) to active tracking velocity ($1.35\,\text{rad/s}$).

### 5. Interactive Particle FX & Kinematic Beams
* **Holographic Beams**: When pinching, a procedural energy filament extends from the fingertip to the core. A unit cylinder is reoriented along the connecting vector via quaternion alignment and dynamic radius expansion without recreating geometry.
* **Comet Hand Trails**: Ring-buffered particle emitter tracking the hand's metric velocity; emits glowing sparks with velocity-dependent lifespans and brightness.
* **Particle Shockwave Bursts**: Fist contraction concentrates energy inward; upon release, an expanding ring accompanied by hundreds of high-velocity ballistic sparks detonates radially.
* **Fingertip Reticle & Target Lock**: Billboarding HUD reticle featuring an inner dot, rotating tick wheel, and corner brackets that transition to amber target lock upon acquiring orbital interactive nodes.

### 6. Cinematic Post-Processing
* **UnrealBloomPass**: Fine-tuned thresholding ($0.62$) so only intense core regions and synaptic pulses trigger the radiant bloom.
* **Composite Lens Grade**:
  * **Radial Chromatic Aberration**: Displaces RGB color channels proportionally to squared distance from the lens center ($r^2$).
  * **Dynamic Film Grain**: High-frequency pseudo-random noise scaled to viewport resolution.
  * **Aerospace Vignette & Scanlines**: Subtle CRT horizontal line modulation.
  * **System Alert Wash**: Amber color grade activated upon tracking interruption or error states.

---

## Vision Pipeline & Human-Computer Interaction

```
[Webcam Feed] ──> [ImageBitmap] ──(Transferable)──> [Dedicated Vision Worker]
                                                              │
                                                   MediaPipe HandLandmarker
                                                              │
[Main Thread] <──(Worker PostMessage)── [Smoothed 3D Landmark Stream]
      │
      ├──> [One Euro Filter] (Low-latency noise rejection)
      ├──> [Hand Scale Normalization] (Wrist-to-Knuckle distance)
      ├──> [Hysteresis Latches] (Chatter elimination)
      └──> [Gesture State Machine] (Debounced intent recognition)
```

1. **Off-Thread Inference**: Video frames are captured as `ImageBitmap` and transferred without memory copying to a dedicated Web Worker. MediaPipe inference never halts or stalls the 60 FPS Three.js render loop.
2. **One Euro Low-Pass Filtering**: Landmark jitter is eliminated using adaptive cutoff frequencies based on instantaneous velocity (Casiez et al., CHI 2012).
3. **Scale-Invariant Feature Extraction**: Distances such as thumb-to-index pinch separation are normalized against the hand's anatomical scale ($d_{\text{wrist} \to \text{knuckle}}$), ensuring gestures remain accurate whether close to or far from the webcam.
4. **Stable Slot Assignment**: A bipartite minimum-cost matcher tracks left and right hands consistently across frames, preventing identity swaps.

---

## Controls

### Hand Gestures

| Gesture | Action | Visual & Audio Response |
| :--- | :--- | :--- |
| **Open Palm** (5 fingers) | Core Activation | Synaptic network brightens, rings accelerate, energy pulse radiates outward |
| **Pinch** (Thumb + Index) | Orb Rotation & Grab | Forms holographic energy filament from fingertip to core; drag to rotate |
| **Fist** | Power Charge | Particle cloud contracts inward, warming to deep gold; release triggers shockwave burst |
| **Two Hands** | Scale & Modulate | Move apart/together to scale orb ($0.62\times - 1.85\times$); left steers, right modulates power |
| **Point** (Index only) | Holographic Cursor | Projects HUD targeting reticle; pinch while hovering an orbital node to select it |
| **Horizontal Swipe** | Configuration Shift | Quick horizontal sweep shifts between procedural configurations |
| **2 / 3 / 4 Fingers** | System Analysis Modes | Triggers dedicated configuration states and diagnostics |

### Mouse & Touch Fallback
When no camera is present, Ultimaitron AI automatically routes pointer events into synthetic `InputHand` data:
* **Move**: Steer the holographic core and position cursor
* **Hold Left Click**: Engage pinch mode to rotate the orb in 3D
* **Scroll Wheel**: Smooth zoom ($3.2 - 14.0$ world units)
* **Touch Drag**: Multi-touch pinch and rotate support on mobile/tablet

### Keyboard Shortcuts

| Key | Function |
| :---: | :--- |
| `1` – `5` | Switch mode: Neural Network, Reactor Core, Data Sphere, Energy Field, Scanner Array |
| `Space` | Detonate manual power pulse and radial shockwave |
| `G` | Toggle Optical Camera input (launches MediaPipe vision pipeline) |
| `R` | System Reset: Re-centers orientation, defaults zoom to 6.2, resets mode to NEURAL |
| `D` or `C` | Toggle Diagnostic Overlay (camera preview, landmark skeleton, FPS, draw calls) |
| `H` | Toggle Interface Visibility (auto-dims HUD panels while keeping controls accessible) |
| `Esc` | Dismiss active modal notice or alert dialog |

---

## Procedural Web Audio Engine

Ultimaitron AI requires no external audio samples. Every sound cue is synthesized procedurally in real-time via the Web Audio API:
* **Ambient Drone**: Sub-bass dual sine oscillators ($55\,\text{Hz}, 82.5\,\text{Hz}, 110\,\text{Hz}$) modulated by a $0.07\,\text{Hz}$ Low-Frequency Oscillator (LFO).
* **Pinch & Release Clicks**: High-frequency square-wave pulses filtered through a sharp bandpass ($Q=6$).
* **Power Surge & Shockwave**: Exponentially down-sweeping sub-bass tone ($120 \to 42\,\text{Hz}$) paired with decaying filtered white noise burst.
* **Mode Shifts & Selections**: Resonant lowpass frequency sweeps ($240 \to 1400\,\text{Hz}$) with logarithmic amplitude envelopes.

---

## Project Structure

```
src/
├── main.ts                  # Application lifecycle, render loop, event coordination
├── config.ts                # Master parameter manifest (every tunable value lives here)
├── vite-env.d.ts            # Vite & GLSL module declarations
│
├── scene/                   # 3D Scene Graph & Rendering
│   ├── RendererFactory.ts   # WebGL context setup, power preference, context loss recovery
│   ├── SceneManager.ts      # Scene initialization, camera rig, lights
│   ├── CameraRig.ts         # Parallax dampening, figure-8 organic drift, aspect ratio adaptation
│   ├── Background.ts        # Atmospheric starfield sphere & core backing haze
│   └── PostProcessing.ts    # EffectComposer, UnrealBloomPass, custom composite shader
│
├── orb/                     # Core Holographic Holography
│   ├── NeuralOrb.ts         # Master controller: modes, envelopes, rotations, pulses
│   ├── OrbModes.ts          # Procedural coordinate generators for 5 configurations
│   ├── OrbUniforms.ts       # Shared uniform buffer maps between points & synapses
│   ├── ParticleSystem.ts    # 20k particle THREE.Points draw call
│   ├── NeuralNetwork.ts     # Synapse lines THREE.LineSegments draw call
│   ├── EnergyCore.ts        # 6-layer inner reactor assembly & billboard ray fan
│   ├── OrbitalRings.ts      # 9 asymmetric holographic concentric rings
│   └── DataMotes.ts         # Orbiting data motes and drifters
│
├── vision/                  # Vision & Optical Tracking
│   ├── MediaPipeManager.ts  # Web Worker bridge with main-thread fallback
│   ├── vision.worker.ts     # Dedicated module worker running HandLandmarker
│   ├── CameraFeed.ts        # WebRTC getUserMedia management & permission handling
│   ├── LandmarkSmoother.ts  # One Euro filter pipeline (21 landmarks x 3 axes)
│   ├── GestureEngine.ts     # Feature extraction, joint angles, size normalization
│   ├── HandTracker.ts       # Multi-hand slot assignment & tracking state machine
│   └── types.ts             # Landmark indices, connections, types
│
├── interaction/             # Human-Computer Interface
│   ├── HandInteraction.ts   # Screen-space to 3D world projection & physics damping
│   ├── GestureStateMachine.ts # Debounced gesture transition state machine
│   ├── InputAdapter.ts      # Normalized input mapping
│   ├── MouseFallback.ts     # Pointer stand-in for webcam-free operation
│   └── InputTypes.ts        # Unified hand telemetry structures
│
├── effects/                 # Real-time Visual Effects
│   ├── HolographicBeam.ts   # Energy beam from hand to core
│   ├── HandTrail.ts         # Comet tail point cloud with charge intensity
│   ├── ParticleBurst.ts     # Radial shockwave ring & spark emitter
│   ├── Reticle.ts           # Billboarding holographic fingertip HUD cursor
│   ├── ScanArc.ts           # Sweeping targeting scanner with angle latching
│   ├── OrbitalNodes.ts      # Selectable interactive holographic nodes
│   └── HandSkeleton.ts      # 3D joint and bone visualizer for debug overlay
│
├── shaders/                 # Custom GLSL Shaders
│   ├── common.glsl          # Shared displacement, morph weights, pulse wavefronts
│   ├── noise.glsl           # Inlined Simplex 3D noise & curl algorithms
│   ├── particle.vert/frag   # Dynamic neural cloud point shader
│   ├── neural.vert/frag     # Synapse connection line shader
│   ├── core.vert/frag       # Procedural inner plasma & Fresnel outer shell
│   ├── hologram.vert/frag   # Procedural tick dashes, sweeps & rings
│   ├── beam.vert/frag       # Energy beam cylinder shader
│   ├── burst.vert/frag      # Ballistic spark physics shader
│   ├── motes.vert/frag      # Orbital data particle motion shader
│   ├── trail.vert/frag      # Hand comet trail shader
│   ├── composite.vert/frag  # Aberration, grain, scanlines, vignette, alert grade
│   └── index.ts             # Shader inclusion & preprocessing registry
│
├── ui/                      # Aerospace HUD & DOM Instrumentation
│   ├── HUD.ts               # Telemetry panels, real-time readouts, meter widgets
│   ├── SystemStatus.ts      # Header brand, connection status, control buttons
│   ├── GestureDisplay.ts    # Center confirmations, mode banner, legend
│   ├── DebugPanel.ts        # 2D landmark overlay, canvas monitor, engine metrics
│   ├── BootSequence.ts      # Cold-start boot choreography & skip handling
│   ├── dom.ts               # Declarative DOM builder primitives
│   └── styles.css           # Instrument styling, glitch animations, theme variables
│
├── audio/                   # Procedural Web Audio Engine
│   └── AudioEngine.ts       # Zero-sample procedural synthesizer
│
└── utils/                   # Mathematics & Hardware Architecture
    ├── MathUtils.ts         # Lerp, damp, OneEuroFilter, Hysteresis, Debouncer
    ├── Noise.ts             # CPU Simplex noise & curl routines
    ├── SpatialHash.ts       # 3D uniform grid spatial hash for O(N) connections
    ├── DeviceCapabilities.ts# GPU detection, SwiftShader identification, tier selection
    ├── PerformanceManager.ts# Adaptive quality engine with anti-oscillation dwell
    └── EventBus.ts          # Strictly typed pub/sub event bus
```

---

## Performance Tiers & Adaptive Engine

To guarantee a fluid frame rate across everything from ultrabooks to high-end workstations, Ultimaitron AI evaluates hardware capability and dynamically adjusts quality:

| Metric | HIGH Tier | MEDIUM Tier | LOW Tier |
| :--- | :---: | :---: | :---: |
| **Neural Particles** | 20,000 | 10,000 | 5,000 |
| **Synapse Connections** | 8,000 | 4,000 | 2,000 |
| **Data Motes** | 1,400 | 800 | 400 |
| **Bloom Strength / Radius** | 0.62 / 0.42 | 0.58 / 0.40 | 0.50 / 0.38 |
| **Pixel Ratio Cap** | 2.0 | 1.5 | 1.0 |
| **Chromatic Aberration** | Enabled | Enabled | Disabled |
| **Procedural Film Grain** | Enabled | Enabled | Disabled |

* **Hysteresis Safeguard**: If frame rate drops below 45 FPS over a 90-frame window, the engine steps down one tier. It requires a sustained 10 seconds above 58 FPS plus a 6-second cooldown before permitting an upgrade, preventing performance thrashing.

---

## Quick Start & Verification

### Prerequisites
* Node.js 18+
* Modern browser with WebGL 2.0 support (Chrome, Edge, Firefox, Safari)

```bash
# 1. Clone & install dependencies
git clone https://github.com/kasanagottu-sathvik/ULTIMAITRON-UI.git
cd ULTIMAITRON-UI-main
npm install

# 2. Run local development server
npm run dev
# Open http://localhost:5173

# 3. Verify TypeScript types & production build
npm run typecheck
npm run build
npm run preview
```

> [!NOTE]
> Camera access requires a secure origin (`https://` or `localhost`). Insecure HTTP deployments will gracefully fall back to pointer control with status alerts.

---

## Technology Stack & Attribution

* **Rendering Engine**: [Three.js](https://threejs.org/) (MIT)
* **Computer Vision**: [Google MediaPipe Tasks Vision](https://developers.google.com/mediapipe) (Apache 2.0)
* **Noise Algorithms**: Simplex Noise by Stefan Gustavson & Ashima Arts (MIT)
* **Filter Mathematics**: 1€ Filter by G. Casiez, N. Roussel, D. Vogel (CHI 2012)
* **Build System**: [Vite](https://vitejs.dev/) & [TypeScript](https://www.typescriptlang.org/)

---

## License

This project is licensed under the [MIT License](LICENSE).
All 3D geometry, custom GLSL shaders, procedural audio graphs, and UI designs are original to Ultimaitron AI.
