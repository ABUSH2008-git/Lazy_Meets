# Demo video plan (about 4 minutes)

Record the screen with OBS, or Win + Alt + R on Windows (Xbox Game Bar). Use your real sample recording, not the synthetic demo, so the video matches the sample outputs you submit.

| Time | Show | Say (roughly) |
|---|---|---|
| 0:00 | The app's home page | "This is LazyMeets. It turns a meeting recording into a transcript, a corrected transcript, and minutes with decisions and action items." |
| 0:15 | Sidebar → *API key & models* open | "Three models: Whisper large-v3 for speech-to-text, GPT-OSS 20B to fix misheard terms, and GPT-OSS 120B to write the minutes. All on Groq." |
| 0:30 | Upload `team_sync.m4a`, open *Help the AI…*, fill participants + glossary + domain, turn on speaker detection | "I add the names and a few terms. That helps Whisper and the refinement model spell them right." |
| 0:50 | Click **Process meeting**; let the three stages tick through | "Each stage shows its status live." |
| 1:30 | *Transcripts* tab, side by side | Point at one or two red→green fixes. "Raw on the left, refined on the right. Only the misheard terms changed. Timestamps are the same." |
| 1:50 | *Refinement changes* tab | Show a blocked row if there is one. "Any fix that would change a number, a 'not', a commitment, or a name gets blocked." |
| 2:10 | *Speakers* tab → name the speakers → *Save names & regenerate minutes* | "Speaker detection runs locally. Once I name them, 'I'll do it' gets tied to the right person." |
| 2:35 | *Meeting record* tab: decisions, then action items | "Only agreed things are decisions. The JAX idea is under proposals not agreed. The README task has no owner because nobody took it, so it says Unspecified." Click a ▶ timestamp to play that moment. |
| 3:05 | Scroll to suggested follow-ups + automatic checks | "'Maybe Ananya could help' isn't a task, because nobody confirmed it. Here's what the checks changed." |
| 3:20 | *Ask the meeting*: "Who is doing the ONNX export and by when?" | Show the answer with its timestamp. |
| 3:35 | *Downloads* tab → **Download all**; open `meeting_record.md` and `meeting_record.json` briefly | "Markdown for people, JSON for machines, same decisions and tasks in both." |
| 3:50 | Upload `notes.txt` or an empty file | "Bad files get a clear error." |
| 4:00 | End | |

Tips: zoom the browser to 110% so text is readable, close other tabs, and do one dry run first so nothing loads for the first time on camera.
