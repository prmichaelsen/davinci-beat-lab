# SceneCraft Platform Architecture

**Concept**: Cloud platform architecture for SceneCraft (scenecraft.online) — AI music video generation as a service with per-customer cloud desktops
**Created**: 2026-03-26
**Status**: Proposal

---

## Overview

SceneCraft evolves from a local CLI tool (davinci-beat-lab) into a cloud platform where customers get a provisioned cloud desktop running the SceneCraft server. The GUI timeline editor (beatlab-synthesizer / scenecraft-synthesizer) connects to their desktop via the REST API. GPU compute (Vast.ai) and AI APIs (Vertex AI / AI Studio) are provisioned under our account and billed to customers with markup.

Projects are YAML files on mounted volumes — portable, concrete, no database required.

---

## Problem Statement

- Local-only tooling limits audience to technical users comfortable with CLI, Python, SSH
- Ephemeral servers lose state between sessions — render caches, stems, styled frames are expensive to regenerate
- Multi-tenant servers create isolation headaches (filesystem conflicts, resource contention, security)
- Users need persistent workspace with their project files, cached intermediates, and stems readily available
- Direct Vast.ai / AI Studio account management is too complex for end users

---

## Solution

**One cloud desktop per customer.** Each customer gets a lightweight VM with:

- Mounted persistent volume storing all project files (YAML, work dirs, stems, cached renders)
- SceneCraft server running on the VM, exposing REST endpoints
- Web GUI (scenecraft-synthesizer) served from the VM or a CDN, connecting to the server
- Vast.ai GPU instances provisioned on-demand from our master account when the customer needs rendering/separation
- AI API calls (Vertex AI / AI Studio) routed through our account with credit-based billing

### Why not alternatives?

| Approach | Rejected Because |
|---|---|
| Ephemeral servers | Lose cached stems, styled frames, render intermediates — regeneration costs $$ and hours |
| Multi-tenant server | Filesystem isolation, resource contention, security complexity, noisy neighbor problems |
| Local-only | Limits audience to technical users, no collaboration, no mobile access |
| GCS / cloud object storage | Adds latency for IO-heavy operations, YAML files work best as local filesystem |
| SQL database | Over-engineered for project data that's naturally document-shaped (YAML) |

---

## Implementation

### Architecture

```
Customer Browser
       │
       │ HTTPS
       ▼
SceneCraft Web GUI (CDN or VM-served)
       │
       │ REST API (authenticated)
       ▼
Customer Cloud Desktop (lightweight VM)
  ├── SceneCraft Server (Python, port 8888)
  │     ├── REST endpoints (project CRUD, render, analyze, effects, etc.)
  │     ├── WebSocket (progress streaming, live preview)
  │     └── Static file serving (styled images, preview clips)
  ├── Mounted Volume (/data)
  │     ├── projects/
  │     │     └── my-video/
  │     │           ├── narrative_keyframes.yaml
  │     │           ├── .beatlab_work/
  │     │           │     ├── audio.wav
  │     │           │     ├── stems/ (drums.wav, bass.wav, vocals.wav, other.wav)
  │     │           │     ├── beats.json
  │     │           │     ├── google_styled/
  │     │           │     ├── google_segments/
  │     │           │     └── ...
  │     │           └── ingredients/ (character/object refs)
  │     └── assets/ (uploaded source videos/audio)
  └── Config
        ├── .scenecraft/credentials.yaml (encrypted API keys)
        └── .scenecraft/billing.yaml (credit balance, usage tracking)

Customer Cloud Desktop
       │
       │ SSH (automated, from server code)
       ▼
Vast.ai GPU Instance (our account)
  ├── Demucs stem separation
  ├── Future: local SD/Flux rendering
  └── Provisioned on-demand, destroyed after use

Customer Cloud Desktop
       │
       │ HTTPS API calls (our account keys)
       ▼
Google AI APIs
  ├── Vertex AI / AI Studio (Nano Banana, Veo 3.1, Gemini)
  └── Billed per-call, tracked against customer credit balance
```

### Component 1: Customer Cloud Desktop

**Provider**: DigitalOcean, Hetzner, or similar (avoid irreversible scale-ups)

**Base spec**:
- 2 vCPU / 4GB RAM (SceneCraft server + light processing)
- 50GB boot disk
- Mounted volume: 100GB default (expandable)
- Ubuntu 24.04 LTS

**Lifecycle**:
- **Provisioned** on signup — customer gets a desktop within minutes
- **Active** — VM running, server accessible, volume mounted
- **Hibernated** — VM stopped after inactivity (configurable, e.g. 30 min). Volume persists, compute released. Storage-only billing.
- **Wake** — VM restarted on next request. Seconds to resume.
- **Destroyed** — on account deletion. Volume snapshot offered before destruction.

**Scaling**:
- Customers can upgrade CPU/RAM for local processing (librosa analysis, OpenCV effects, ffmpeg encoding)
- Heavy GPU work always goes to Vast.ai — the desktop never needs a GPU
- Volume size expandable via billing portal

### Component 2: Vast.ai GPU Management

**Account model**: All instances provisioned under our Vast.ai master account.

**Per-customer isolation**:
- Each customer's GPU work tagged with customer ID
- Remote work directory: `/workspace/{customer_id}/`
- Instance state stored on customer's volume: `.scenecraft/vast_instance.json`
- Instances destroyed after job completion (or configurable keep-alive)

**Billing**:
- Track GPU-hours per customer
- Bill at markup over Vast.ai cost (e.g. 1.5x)
- Usage visible in customer billing dashboard

**Instance types**:
- **Stems** (Demucs): 8GB VRAM, cheap, ~$0.15/hr
- **Rendering** (future SD/Flux): 16GB+ VRAM, ~$0.50-1.00/hr
- Customer doesn't choose — SceneCraft auto-provisions the right type per job

### Component 3: AI API Credit System

**Model**: Prepaid credits purchased by customer, debited per API call.

**Tracked operations**:
| Operation | Approximate Cost | Credit Debit |
|---|---|---|
| Nano Banana stylize (per image) | ~$0.01 | 1 credit |
| Veo 3.1 video generation (8s clip) | ~$0.10 | 10 credits |
| Gemini transition description | ~$0.005 | 0.5 credits |
| Claude AI effect plan | ~$0.02 | 2 credits |
| Demucs stem separation (per track) | GPU-hour based | variable |

**Implementation**:
- Credit balance stored in `.scenecraft/billing.yaml` on customer volume
- Server checks balance before expensive operations
- Low-balance warnings at 10% remaining
- Auto-top-up optional

### Component 4: Project File Format

Projects are **YAML on disk** — no database.

```
projects/my-music-video/
  ├── project.yaml              # Project metadata, settings
  ├── narrative_keyframes.yaml  # Timeline: keyframes, transitions, sections
  ├── ingredients/              # Character/object reference images
  │     ├── protagonist.png
  │     └── vehicle.png
  ├── .beatlab_work/            # Cached intermediates (auto-managed)
  └── exports/                  # Final rendered outputs
```

**Benefits of YAML over SQL**:
- Projects are portable — zip and download, share via email, move between desktops
- Human-readable — customers can inspect/edit in any text editor
- Version-controllable — git-friendly diffs
- No migration headaches — schema changes are just new optional fields
- Concrete like a `.docx` file — the project IS the files

### Component 5: REST Server (extends local.beatlab-server.md)

The existing `beatlab server` design applies directly. On the cloud desktop, the server starts on boot and stays running. Additional platform endpoints:

```
# Platform management
GET  /api/billing/credits           # Current credit balance
POST /api/billing/purchase          # Buy more credits
GET  /api/billing/usage             # Usage history

GET  /api/gpu/status                # Active GPU instances
POST /api/gpu/destroy               # Destroy a GPU instance

GET  /api/storage/usage             # Volume usage
POST /api/storage/expand            # Request more storage

# All existing beatlab server endpoints carry over:
GET  /api/projects
GET  /api/projects/:id/keyframes
POST /api/projects/:id/analyze
POST /api/projects/:id/render
POST /api/projects/:id/effects
...
```

---

## Benefits

- **Isolation**: Each customer has their own filesystem, processes, and cached data — no cross-contamination
- **Persistence**: Stems, styled frames, and render caches survive between sessions — no re-processing
- **Simplicity**: No database, no object storage, no multi-tenant security model — just files on a disk
- **Portability**: YAML projects can be downloaded, shared, version-controlled
- **Scalable compute**: Lightweight base desktop + on-demand GPU = pay for what you use
- **Familiar model**: "Your computer in the cloud" is intuitive for creative professionals

---

## Trade-offs

- **Base cost per customer**: Even a hibernated VM costs ~$2-5/month in storage. Need pricing to cover this baseline. Mitigated by hibernation and minimum plan pricing.
- **Cold start on wake**: Hibernated VMs take 5-30 seconds to resume. Mitigated by keeping VMs active during active sessions and only hibernating after extended inactivity.
- **No real-time collaboration**: One desktop = one user. Mitigated by export/share workflows. Future: add collaboration via shared volumes or project sync.
- **Operational burden**: Managing per-customer VMs at scale requires automation. Mitigated by infrastructure-as-code (Terraform/Pulumi) and a management API.
- **Vast.ai reliability**: Third-party GPU provider may have availability issues. Mitigated by fallback to alternative providers, queue system for GPU jobs.

---

## Dependencies

- **Cloud VM provider**: DigitalOcean, Hetzner, or similar (with API for automated provisioning)
- **Vast.ai**: GPU compute (master account with API key)
- **Google AI APIs**: Vertex AI or AI Studio (master account with billing)
- **Anthropic API**: Claude for AI effect planning, transition descriptions
- **Domain**: scenecraft.online
- **Auth provider**: TBD (email/password, OAuth, or API keys)
- **Payment processor**: Stripe (credit purchases, subscription billing)
- **local.beatlab-server.md**: REST server design (extends this)

---

## Testing Strategy

- **Provisioning**: Automated test that creates a desktop, runs a project, and destroys it
- **Hibernation**: Verify volume persists across stop/start cycles
- **GPU isolation**: Verify customer A's Vast.ai jobs don't access customer B's files
- **Credit tracking**: Verify API calls are metered correctly and balance decrements match
- **Billing**: End-to-end test of credit purchase → API usage → balance deduction
- **Cold start**: Measure wake time from hibernation under various volume sizes

---

## Migration Path

1. **Phase 1**: Continue running locally (current state). SceneCraft server works on localhost.
2. **Phase 2**: Manual cloud desktop provisioning. Spin up a VM, install SceneCraft, mount volume. Test with 1-2 users.
3. **Phase 3**: Automated provisioning API. Signup → VM created → server running → GUI accessible.
4. **Phase 4**: Billing integration. Credit purchases, usage tracking, hibernation policies.
5. **Phase 5**: Public launch on scenecraft.online.

---

## Key Design Decisions

### Infrastructure

| Decision | Choice | Rationale |
|---|---|---|
| Per-customer isolation | Cloud desktop (VM) per customer | Avoids multi-tenant complexity, gives persistent filesystem |
| Storage | Mounted volumes, no GCS/S3 | IO-heavy operations need local-speed access; YAML files are filesystem-native |
| Data format | YAML on disk, no SQL | Projects are document-shaped, portable, human-readable, git-friendly |
| GPU provisioning | Our Vast.ai account, billed to customer | Simpler UX — customer doesn't need their own Vast.ai account |
| AI API routing | Our API keys, credit-based billing | Unified billing, no customer API key management |
| VM scaling | Lightweight base + external GPU | Avoids irreversible scale-ups; GPU work is bursty, not sustained |

---

## Future Considerations

- **Collaboration**: Shared project volumes or real-time sync for team workflows
- **Mobile app**: Lightweight viewer/approver that connects to customer's desktop
- **Marketplace**: Shared ingredients (characters, style packs) that customers can purchase
- **Self-hosted option**: Allow power users to run SceneCraft on their own hardware
- **Feature film pipeline**: Consistent characters (Veo 3.1 Ingredients), dialogue, shot composition
- **White-label**: Let studios rebrand SceneCraft for their own teams

---

**Status**: Proposal
**Recommendation**: Validate base VM cost model, prototype automated provisioning with one cloud provider, then build Phase 2 (manual cloud desktop for testing)
**Related Documents**: [local.beatlab-server.md](local.beatlab-server.md)
