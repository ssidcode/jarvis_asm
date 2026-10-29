# Running Jarvis in the background

| Platform | File | Notes |
|---|---|---|
| Linux (systemd user unit) | `jarvis.service` | needs `XDG_RUNTIME_DIR` for PipeWire/Pulse audio; `loginctl enable-linger` to survive logout |
| macOS (LaunchAgent) | `com.jarvis.assistant.plist` | must be a **LaunchAgent**, not a LaunchDaemon — daemons have no audio session and cannot get mic permission |
| Raspberry Pi | `jarvis.service` | use `stt.model: tiny.en` and `wakeword` only; see the tuning table in the main README |
| Windows | — | use Task Scheduler: *At log on* → `pythonw.exe -m jarvis`, start-in = repo root |

Put secrets in `~/.jarvis/env` (mode `600`) rather than in the unit file:

```
ANTHROPIC_API_KEY=sk-ant-...
```
