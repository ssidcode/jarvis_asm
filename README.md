# Jarvis — a voice assistant that answers only to you

Continuously listens in the background. Wakes on **"Hey Jarvis"**. Verifies the
voice against *your* enrolled voiceprint before it does anything, so if someone
else says the wake word, nothing happens.

```
mic ──► ring buffer ──► wake word ──► ① voice check ──► record ──► ② voice check
        (3 s lookback)   "Hey Jarvis"   (wake phrase)     (VAD)      (command)
                                             │                          │
                                        reject ✗                   reject ✗
                                                                        ▼
                                              speak ◄── LLM ◄── Whisper STT
```

## What "trained on my voice" means here

Worth being precise, because the obvious approach is the wrong one.

**This does not fine-tune a model on your voice.** A model trained only on
recordings of you has never seen a counter-example, so it learns "this is
speech", not "this is *Ishan*", and it will happily accept a stranger.

Instead it uses **speaker verification**: a pretrained ECAPA-TDNN encoder
(trained on VoxCeleb — thousands of speakers) maps any utterance to a
192-dimensional embedding in a space where distance means *different person*.
You enroll once; your clips become a stored voiceprint. At runtime, each clip is
embedded and compared by cosine similarity. Above your threshold, it's you.

That gives you a tunable, measurable security boundary — you can calculate your
false-accept rate — which a fine-tune cannot.

**Verification runs twice per turn**, which matters more than it sounds:

1. On the ~1.3 s containing the wake phrase — a cheap early reject, so a
   stranger never even starts a recording.
2. On the full command utterance — the real gate, with more audio to judge.

Stage 2 exists because stage 1 alone is trivially defeated: you say "Hey
Jarvis", someone else immediately says "transfer the money." Stage 1 passes.
Stage 2 is what catches it. (There's a test for exactly this.)

---

## Install

Python **3.11 is recommended** (see the version note in `requirements.txt`);
3.9–3.13 all work.

```bash
git clone <your-repo> jarvis && cd jarvis
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**System audio libraries** (`sounddevice` needs PortAudio):

| OS | Command |
|---|---|
| Debian/Ubuntu/Pi | `sudo apt install portaudio19-dev libsndfile1 ffmpeg` |
| Fedora | `sudo dnf install portaudio-devel libsndfile ffmpeg` |
| macOS | `brew install portaudio libsndfile ffmpeg` |
| Windows | nothing — wheels are self-contained |

**Your API key.** Groq is the default for both the LLM and speech-to-text:

```bash
export GROQ_API_KEY=gsk_...          # console.groq.com/keys
```

Groq is a good fit here — voice assistants live or die on latency, and it's the
fastest hosted option. Using it for **both** stages is configured by default:

```yaml
llm:
  backend: groq
  model: llama-3.3-70b-versatile     # or llama-3.1-8b-instant for lowest latency
stt:
  backend: groq
  groq_model: whisper-large-v3-turbo
```

Putting STT on Groq too is the bigger win: it replaces local Whisper, so there's
no ~500 MB model download and transcription drops from 0.3–1.5 s to roughly
150–350 ms. The trade-off is that your command audio leaves the machine. To keep
speech local while still using Groq's LLM, set `stt.backend: faster-whisper`.

Current Groq production models (check
[console.groq.com/docs/models](https://console.groq.com/docs/models), as IDs get
deprecated):

| Model | Use |
|---|---|
| `llama-3.3-70b-versatile` | best general quality (default) |
| `llama-3.1-8b-instant` | lowest latency; fine for short commands |
| `openai/gpt-oss-120b` | strongest reasoning |
| `whisper-large-v3-turbo` | STT (default) |

If the key is missing or Groq is unreachable, STT **falls back to local Whisper
automatically** rather than failing the turn. The LLM does not fall back — it
tells you the key is missing, because silently switching your brain isn't a
decision this should make for you.

Other backends: `anthropic` (needs `ANTHROPIC_API_KEY`), or `ollama` for fully
local — see *Fully offline* below.

### Setup, in order

```bash
# 0. check every stage and see what's missing before anything else
python -m scripts.diagnose

# 1. find your microphone and check your levels
python -m scripts.list_devices --test        # want peaks around -20 to -10 dBFS
#    put the device number in config.yaml -> audio.input_device

# 2. download a TTS voice (Linux; macOS can use the built-in `say`)
python -m scripts.fetch_voice

# 3. enroll your voice — 8 prompted phrases, ~1 minute
python -m scripts.enroll

# 4. calibrate your threshold against real data instead of guessing
python -m scripts.calibrate --write

# 5. run it
python -m jarvis
```

Then say **"Hey Jarvis"**, wait for the beep, and talk.

```bash
python -m jarvis --check        # load every stage and exit; good for debugging
```

---

## Step 4 is the one people skip — don't

The default `speaker.threshold: 0.45` is a *starting guess*. The right value
depends on your voice, your mic and your room. Calibration measures it:

```bash
# Best: give it recordings of other people, so false-accept rate is measurable.
# A few wavs of housemates, or clips pulled from a podcast, both work.
python -m scripts.calibrate --impostor-dir ./other_voices --write
```

It sweeps the threshold, prints false-accept / false-reject at each point and
the Equal Error Rate, then picks the lowest threshold that accepts **zero**
impostors while still rejecting you less than 10% of the time.

Without impostor audio it can only measure your own score distribution and
picks a margin below your worst genuine score — safe against locking *you* out,
but it cannot tell you how secure it is.

**Tuning the trade-off:**

| Symptom | Fix |
|---|---|
| It ignores me sometimes | lower `speaker.threshold` by 0.05; re-enroll with `--append` in that room |
| It answered someone else | raise `speaker.threshold` by 0.05 and re-run calibration |
| It triggers on the TV | raise `wakeword.threshold` to 0.6 and `trigger_level` to 2 |
| It misses the wake word | lower `wakeword.threshold` to 0.4 |
| It cuts me off mid-sentence | raise `vad.silence_ms` to 1100 |
| It waits too long after I finish | lower `vad.silence_ms` to 600 |

---

## How the pipeline works

| Stage | Module | What it does | Cost |
|---|---|---|---|
| Capture | `audio_io.py` | 16 kHz mono int16, 80 ms frames, PortAudio callback → queue | ~0 |
| Lookback | `audio_io.py` | 3 s ring buffer, written on *every* frame | ~0 |
| Wake word | `wakeword.py` | openWakeWord `hey_jarvis` + trigger-level and refractory logic | ~1 ms/frame |
| Voice gate | `speaker_id.py` | ECAPA-TDNN embedding, cosine vs voiceprint | ~80 ms |
| Endpointing | `vad.py` | webrtcvad/silero/energy + the "have you stopped talking" state machine | ~1 ms/frame |
| STT | `stt.py` | Groq Whisper (hosted) or faster-whisper int8 (local) | 0.15–0.35 s / 0.3–1.5 s |
| Brain | `llm.py` | Groq / Anthropic / Ollama / any OpenAI-compatible, streaming | 0.3–2 s |
| Voice | `tts.py` | Piper / `say` / pyttsx3, **sentence-streamed** | 0.2 s to first word |

Two design choices worth knowing about:

**The ring buffer is why wake-word verification is possible at all.** By the
time the detector fires, "Hey Jarvis" is already in the past. The buffer lets us
reach back and run the voiceprint check on the exact audio that triggered us.

**Replies are spoken sentence by sentence as the LLM generates them.** Waiting
for the full response before speaking roughly doubles perceived latency. The
streaming speaker flushes each *completed* sentence (abbreviation-aware, so
"Dr. Smith" doesn't split) while the model is still writing.

### Conversation flow

After a reply there's a **6 s follow-up window** where you can just keep
talking — no wake word needed — but the voice gate still applies. Say any of
`stop`, `never mind`, `cancel`, `that's all` to stand down and clear history.

**Barge-in** is on by default: say "Hey Jarvis" while it's talking and it stops
mid-sentence. The interrupting voice is verified too, so the TV can't cut it off.
With `barge_in: false` it goes half-duplex instead — the mic is muted while
speaking, so it can never hear itself.

---

## Fully offline

No API calls, nothing leaves the machine:

```yaml
llm:
  backend: ollama
  model: llama3.1:8b
  base_url: http://localhost:11434/v1
tts:
  backend: piper
stt:
  model: small.en
```

```bash
ollama serve & ollama pull llama3.1:8b
```

Wake word, voice verification, STT and TTS were already local — only the LLM
reached out.

---

## Run it in the background

See [`service/README.md`](service/README.md). Short version: `jarvis.service`
for a systemd user unit on Linux, `com.jarvis.assistant.plist` for a macOS
LaunchAgent.

Two traps that cost people an afternoon:

* On macOS it **must be a LaunchAgent, not a LaunchDaemon** — daemons have no
  audio session and can never get microphone permission.
* On Linux the unit needs `XDG_RUNTIME_DIR=/run/user/%U` to reach
  PipeWire/PulseAudio, and `loginctl enable-linger $USER` to survive logout.
* **A key exported in your shell is invisible to a systemd/launchd service.**
  Put `GROQ_API_KEY=gsk_...` in `~/.jarvis/env` (mode `600`); the unit reads it
  via `EnvironmentFile`.

### Raspberry Pi / low-power boxes

```yaml
stt:     {backend: groq}            # offload STT entirely; big win on a Pi
llm:     {backend: groq, model: llama-3.1-8b-instant}
vad:     {backend: webrtc}
speaker: {verify_wakeword: false}   # saves ~80 ms; stage 2 still guards you
```

A Pi 4 handles the wake word and speaker verification comfortably. Offloading
STT and the LLM to Groq leaves it doing only the always-on local work, which is
exactly what it's good at. If you'd rather keep speech on-device, use
`stt: {backend: faster-whisper, model: tiny.en}` — roughly real time on a Pi 4.

---

## Security, honestly

What this does and does not protect against:

**Handled.** A stranger saying the wake word. A stranger taking over after your
wake word (stage 2). Clips too short to carry identity — refused, not guessed.
The voiceprint is written `0600`, because it is a credential.

**Not handled.** A good *recording of you* will pass — this is biometric
verification, not liveness detection, and it has no replay defence. If you need
that, add a challenge-response ("Jarvis, repeat: four seven two") or a
presentation-attack-detection model. A determined voice clone will also pass;
current anti-spoofing research is an arms race and none of it is solved.

Treat the voice gate as "my assistant ignores other people in the room", which
it does well — not as an authentication boundary for anything that matters.
Don't wire it to your bank. If you add actions with real consequences, put a
second factor on those specific actions.

Also: **the mic is always on.** Audio stays in a 3-second rolling in-memory
buffer and is never written to disk unless you set `logging.save_audio: true`
(which exists for threshold debugging — it writes the clips Jarvis accepted and
rejected). Nothing leaves the machine before the wake word fires; after it
fires, the transcript goes to whichever LLM backend you configured.

---

## Tests

```bash
python tests/test_pipeline.py        # or: python -m pytest tests/ -v
```

Nine tests drive the real state machine with synthetic audio and every heavy
backend stubbed — no mic, no models, no network. They cover the owner being
answered, a stranger rejected at stage 1, a stranger rejected at stage 2,
silence timing out, stop phrases, follow-up turns, barge-in, half-duplex muting,
and short-clip refusal.

Two real bugs these caught during development, both still regression-tested:

* The energy VAD computed its noise floor from a history that included speech,
  so during sustained speech the floor climbed to the speech level and **cut the
  speaker off mid-sentence**. Now it uses an asymmetric tracker — fast down
  toward quiet, very slow up.
* The endpointer could **hang for the full 20 s** `max_seconds` if a cough
  started capture but real speech never followed, because the `done` condition
  required a speech-frame minimum that was never met. It now abandons early.

---

## Customising

**A different wake word.** `hey_jarvis` is pretrained and free. For a custom
phrase, train one with
[openWakeWord's training notebook](https://github.com/dscripka/openWakeWord)
(~1 hour on free Colab, synthetic data, no recording needed), drop the `.onnx`
in and set `wakeword.model` to its path.

**Jarvis speaking in your voice.** Separate problem from recognising you —
needs a TTS voice clone, not a voiceprint. Piper can be fine-tuned on ~30
minutes of your speech, or use XTTS-v2 for few-shot cloning.

**Giving it actions** (timers, lights, searches). Add tool/function calling in
`llm.py`: pass a `tools=` list, handle `tool_use` blocks in the stream, and run
the handler before speaking the result. Put a confirmation step on anything
irreversible.

## Layout

```
config.yaml            all tuning in one place; every key overridable via
                       JARVIS_<SECTION>_<KEY> env vars
jarvis/
  assistant.py         the state machine that orchestrates everything
  audio_io.py          mic, ring buffer, playback, format conversion
  wakeword.py          openWakeWord wrapper (handles 0.4.x-0.6.x, self-healing
                       model cache, runtime-aware model format)
  speaker_id.py        ECAPA embeddings, voiceprint, the verification gate
  vad.py               three VAD backends + utterance endpointing
  stt.py               faster-whisper (local) or Groq Whisper (hosted)
  llm.py               Groq / Anthropic / OpenAI-compatible, streaming, history
  tts.py               Piper / say / pyttsx3 + sentence-streaming speaker
scripts/
  diagnose.py          check every stage and report what's broken
  enroll.py            record your voiceprint (quality-checked)
  calibrate.py         measure your threshold; --write updates config.yaml
  list_devices.py      pick a mic, verify levels
  fetch_voice.py       download a Piper voice
service/               systemd unit + macOS LaunchAgent
tests/test_pipeline.py end-to-end tests with stubbed backends
```

## Troubleshooting

**Start with `python -m scripts.diagnose`** — it reports versions, which
openWakeWord API and model files are present, whether the voiceprint loads, and
whether each backend can be reached.

| Problem | Cause |
|---|---|
| `unrecognised openwakeword API` | fixed in this version — the loader no longer introspects `Model.__init__` (it's decorated in 0.6.x, so inspection reveals nothing). Upgrade from an earlier copy. |
| `INVALID_PROTOBUF` loading a model | a `.tflite` was handed to onnxruntime. Fixed: model format now follows the available runtime. |
| `NO_SUCHFILE ... hey_jarvis_v0.1.onnx` | half-downloaded model cache. Fixed: missing files are now re-fetched automatically (openWakeWord's own check only looks at the tflite copy, so it never self-heals). |
| `GROQ_API_KEY is not set` | `export GROQ_API_KEY=gsk_...` — and see the note below about services |
| `returned HTTP 401` | bad or expired Groq key |
| `returned HTTP 404` with a model name | that model ID was deprecated; pick a current one from Groq's models page |
| `No voiceprint at ~/.jarvis/voiceprint.npz` | run `python -m scripts.enroll` |
| No audio captured, no error | OS mic permission (macOS: System Settings → Privacy → Microphone, grant it to your *terminal* or the Python binary) |
| `PortAudioError: Invalid device` | run `python -m scripts.list_devices`, set `audio.input_device` |
| Wake word never fires | check levels with `--test`; run with `--log-level DEBUG` to watch per-frame scores |
| It replies to the TV | expected if `speaker.enabled: false`; otherwise raise the threshold and recalibrate |
| `dropped N frames` warnings | CPU saturated — use a smaller `stt.model` |
| Slow first run | model downloads (Whisper ~500 MB, ECAPA ~80 MB), cached afterwards |
