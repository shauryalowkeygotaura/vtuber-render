# Your meme clips (Clash of Clans etc.)

1. Drop the audio here: `hog-rider.mp3`, `goblin-laugh.wav` ... (mp3 / wav / ogg / m4a, short is better, clips are cut at 0.9 s).
2. Copy `memes.example.json` to `memes.json` and list each clip with the words that should trigger it.
3. Preview: `python -m vtuber.soundtrack edits/zero-dollar-v7.json --print`. Your memes show up by name in the cues column.

Rules the loader enforces (bad entries are skipped with a message, never a crash):
- name: `a-z 0-9 _` only
- file: a plain file name in this folder
- words: list of words/phrases (letters, digits, spaces, apostrophes)
- gain_db: -40 to 0 (default -12), loudness of the clip's peak

Your memes beat every built-in sound on the same word.

Audio files in this folder are git-ignored: game audio belongs to its owner (Clash of Clans = Supercell), so it stays local. Before posting a reel that uses them, check Supercell's Fan Content Policy, especially since these reels promote a paid service.
