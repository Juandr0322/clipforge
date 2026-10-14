# Where to get sound effects and music (F48)

ClipForge never downloads audio by itself. Diego picks files, checks the licence, and adds them to his audio library (default `~/clipforge-audio`, or `CLIPFORGE_AUDIO_LIBRARY`) with `clipforge audio add`. Every item records its licence and source URL, so a claim or a client question can be answered from the clip report.

Checked 2026-10-14. Licences change: re-read the licence page when downloading, and keep the source URL.

## Rules of thumb
- Clipping is paid work for clients, so only use licences that allow commercial use. Skip anything marked non-commercial (NC).
- Prefer "no attribution" licences (CC0, Pixabay, Kenney, Sonniss). CC-BY is fine, but the credit line has to go in the post caption; ClipForge puts it in the report.
- One master render is posted on TikTok, Reels and Shorts, so music is used only when its item is cleared for all three (`platforms: ["all"]`). A track limited to one platform is skipped.
- Trending sounds are added by the creator in each app at posting time; they are never baked into the file. Business accounts are limited to each platform's commercial library (TikTok's Commercial Music Library; Meta's Sound Collection on Instagram/Facebook).

## Sound effects
| Source | Licence | Good for | Notes |
|---|---|---|---|
| [Kenney audio packs](https://kenney.nl/assets/category:Audio) | CC0 (no credit, commercial OK) | Impacts, UI pops and clicks, digital and sci-fi hits | "Impact Sounds" has 130 files. Best first download. |
| [Freesound](https://freesound.org) | Per sound: CC0, CC-BY or CC-BY-NC | Whooshes, risers, crowd and meme-style sounds | Use the licence filter. CC0 = no credit; CC-BY = credit title, user, link and licence; never CC-BY-NC. |
| [Sonniss GDC bundle](https://gdc.sonniss.com/) | Royalty-free, commercial use, no attribution, "for media production" | Cinematic whooshes, impacts, risers (the 2026 bundle is 7.47 GB+) | Read the full licence at sonniss.com/gdc-bundle-license before using it in client videos. |
| [Pixabay sound effects](https://pixabay.com/sound-effects/) | Pixabay Content License: free, no attribution | Everything | You may not redistribute the files on their own (so never commit them to the repo). |
| [Mixkit](https://mixkit.co/free-sound-effects/) | Mixkit Sound Effects Free License | Transitions, game sounds | Each item type has its own licence; open the item's licence before adding it. |

## Music (quiet bed under the clip)
| Source | Licence | Notes |
|---|---|---|
| [Pixabay music](https://pixabay.com/music/) | Pixabay Content License: free, no attribution | Keep the track's page URL in case a platform flags it. |
| [Mixkit music](https://mixkit.co/free-stock-music/) | Mixkit Stock Music Free License | Check the item's licence page. |
| [Incompetech (Kevin MacLeod)](https://incompetech.com/music/royalty-free/licenses/) | Free option requires a credit line | Put the credit in `attribution`; it ends up in the clip report. |
| [YouTube Audio Library](https://support.google.com/youtube/answer/3376882) | Safe from Content ID on YouTube and monetizable; CC tracks need credit | YouTube doesn't say these tracks may be used off YouTube, so add them with `--platforms shorts` (ClipForge then skips them for the shared master). |

## Adding a file
```
clipforge audio add path/to/boom.wav --kind sfx --tags impact --license CC0 --source https://kenney.nl/assets/impact-sounds
clipforge audio add path/to/track.mp3 --kind music --tags gaming,energetic --license "Pixabay Content License" --source https://pixabay.com/music/...
clipforge audio check
```
Sound-effect tags the cue rules use: `impact`, `whoosh`, `pop`, `riser`. Any other tags are allowed and kept for later recipes. Music is picked by `CLIPFORGE_MUSIC_TAGS` (default `gaming`), falling back to any cleared track.
