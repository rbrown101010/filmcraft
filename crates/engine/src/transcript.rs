//! Text-based editing commands (`transcript.*`): the Text panel ▸ Transcript tab.
//!
//! Transcripts belong to media items (`Project::transcripts`, media time); the sequence transcript
//! is derived from them ([`filmcraft_edit::transcript::sequence_words`]). Words of the sequence
//! transcript are addressed by index (`from`, `to`, inclusive), as `transcript.inspect` lists them.
//!
//! Speech recognition goes through a [`Transcriber`]: [`Session::transcriber`] when a host or a
//! test installed one, else the Whisper model named by `model` from `<data dir>/models` (needs the
//! engine feature `whisper`; without it `transcript.generate` fails with a clear error, and agents
//! can still bring their own transcript with `transcript.set`).

use std::sync::Arc;

use serde_json::{Value, json};

use filmcraft_edit as edit;
use filmcraft_edit::transcript::{self as tx, CaptionRules, SeqWord};
use filmcraft_project::{CaptionFormat, CaptionTrack, ItemId, ItemKind, TrackId, Transcript};
use filmcraft_speech::{Options, SpeechError, Transcriber};
use filmcraft_time::{TICKS_PER_SECOND, Tick, TimeRange};

use crate::commands::{CommandSpec, always, bad, bool_p, f64_p, has_seq, str_p, u64_p};
use crate::{EngineError, Result, Session};

type Run = fn(&mut Session, &Value) -> Result<Value>;
type Enabled = fn(&Session) -> std::result::Result<(), String>;

fn spec(id: &'static str, label: &'static str, menu: &'static [&'static str], params: &'static str, enabled: Enabled, run: Run, journal: bool) -> CommandSpec {
    CommandSpec { id, label, menu, shortcut: None, params, enabled, run, journal }
}

/// Where downloaded speech models live (`<data dir>/models`).
pub fn models_dir() -> Option<std::path::PathBuf> {
    crate::autosave::default_data_dir().map(|d| d.join("models"))
}

/// Whether this build can transcribe with Whisper (feature `whisper`).
pub fn speech_available() -> bool {
    filmcraft_speech::available()
}

/// Transcripts keyed by both media items and subclips (a subclip inherits its parent media's transcript).
pub fn effective_transcripts(s: &Session) -> tx::Transcripts {
    let mut map = s.project.transcripts.clone();
    for (&id, item) in &s.project.items {
        if matches!(item.kind, ItemKind::Subclip { .. })
            && let Some(m) = media_item(s, id)
            && let Some(tr) = s.project.transcripts.get(&m)
        {
            map.entry(id).or_insert_with(|| tr.clone());
        }
    }
    map
}

/// The words of the active sequence's transcript.
pub fn sequence_words(s: &Session) -> Vec<SeqWord> {
    match s.active_sequence() {
        Some(q) => {
            let map = effective_transcripts(s);
            let mut words = tx::sequence_words(q, &map);
            for w in &mut words {
                if let Some(m) = media_item(s, w.item) {
                    w.item = m;
                }
            }
            words
        }
        None => Vec::new(),
    }
}

fn has_transcript(s: &Session) -> std::result::Result<(), String> {
    has_seq(s)?;
    if sequence_words(s).is_empty() { Err("the sequence has no transcript (Transcribe first)".into()) } else { Ok(()) }
}

fn has_transcripts(s: &Session) -> std::result::Result<(), String> {
    if s.project.transcripts.is_empty() { Err("there are no transcripts".into()) } else { Ok(()) }
}

/// Why speech-to-text can't run in this build (no installed transcriber, built without `whisper`).
pub(crate) const NO_SPEECH: &str =
    "speech-to-text is not available in this build (built without the `whisper` feature); import a transcript with transcript.set instead";

/// `transcript.generate` can run: a host installed a transcriber or the build has speech-to-text
/// (#97: it reported enabled and then always failed).
pub(crate) fn can_transcribe(s: &Session) -> std::result::Result<(), String> {
    if s.transcriber.is_some() || speech_available() { Ok(()) } else { Err(NO_SPEECH.into()) }
}

/// `transcript.downloadModel` can run: built with `speech-download` (#98).
fn can_download(_: &Session) -> std::result::Result<(), String> {
    if cfg!(feature = "speech-download") {
        Ok(())
    } else {
        Err("model downloads are not available in this build (built without the `speech-download` feature)".into())
    }
}

/// The media item behind a project item (subclips resolve to their parent).
pub fn media_item(s: &Session, item: ItemId) -> Option<ItemId> {
    match &s.project.item(item)?.kind {
        ItemKind::Media(_) => Some(item),
        ItemKind::Subclip { parent, .. } => media_item(s, *parent),
        _ => None,
    }
}

fn ids_p(p: &Value, k: &str) -> Option<Vec<ItemId>> {
    p.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(ItemId).collect())
}

/// Items to transcribe: `items` / `item`, else the Project panel selection, else the media of the
/// active sequence's enabled audio clips. Subclips resolve to their media; duplicates are removed.
fn targets(s: &Session, p: &Value) -> Vec<ItemId> {
    let mut raw = ids_p(p, "items").or_else(|| u64_p(p, "item").map(|i| vec![ItemId(i)])).unwrap_or_default();
    if raw.is_empty() {
        raw = s.state.project_selection.clone();
    }
    if raw.is_empty()
        && let Some(q) = s.active_sequence()
    {
        raw = q.audio_tracks.iter().flat_map(|t| t.items.iter()).filter(|it| it.enabled).map(|it| it.item).collect();
    }
    let mut out = Vec::new();
    for i in raw {
        if let Some(m) = media_item(s, i)
            && !out.contains(&m)
        {
            out.push(m);
        }
    }
    out
}

/// Mono 16 kHz audio of a media item (None: no audio).
fn item_audio(s: &Session, item: ItemId) -> Option<Vec<f32>> {
    let dur = match &s.project.item(item)?.kind {
        ItemKind::Media(m) => m.duration(),
        _ => return None,
    };
    let src = s.source(item)?;
    if !src.info().has_audio() {
        return None;
    }
    let sr = filmcraft_speech::SAMPLE_RATE;
    let len = dur.to_units_floor(sr as i64).max(0) as usize;
    let buf = src.audio(0, len, sr).ok()?;
    Some(filmcraft_speech::downmix(&buf.channels))
}

fn speech_err(e: SpeechError) -> EngineError {
    EngineError::Other(e.to_string())
}

/// The transcriber to use: the installed one, else the named catalogue model (or built-in acoustic
/// transcriber when external model weights are not yet downloaded).
fn transcriber(s: &Session, p: &Value) -> Result<Arc<dyn Transcriber>> {
    if let Some(t) = &s.transcriber {
        return Ok(t.clone());
    }
    // Settings ▸ Media Analysis & Transcription ▸ Speech model
    let model = str_p(p, "model").unwrap_or(&s.prefs.media_analysis.whisper_model);
    if !filmcraft_speech::available() {
        return Err(EngineError::Other(NO_SPEECH.into()));
    }
    let dir = models_dir();
    filmcraft_speech::load_or_builtin(dir.as_deref(), model).map_err(speech_err)
}

/// Scene-tailored script turns for the procedural `DemoScene` clips when no custom `s.transcriber`
/// is installed, so transcribing the demo project produces a cohesive documentary dialogue script.
fn demo_scene_transcript(scene: filmcraft_media::DemoScene, audio: &[f32], opts: &Options, source_id: &str) -> Transcript {
    use filmcraft_media::DemoScene::{Aurora, CityNight, Dunes, Forest, OceanSunset, Plasma};
    let (speakers, turns): (&[&str], &[(u32, &str)]) = match scene {
        OceanSunset => (
            &["Narrator", "Director"],
            &[
                (0, "Every great story begins at the edge of the water, um as the golden hour light settles."),
                (1, "Let's hold on this wide coastal frame before we cut into the city night sequence."),
                (0, "The Pacific horizon glows warm amber while the tide rolls steadily across the shore."),
                (1, "Good, uh mark that take for the opening title sequence."),
            ],
        ),
        CityNight => (
            &["Narrator", "Director"],
            &[
                (0, "After dusk the neon reflections turn the wet pavement into a moving canvas."),
                (1, "Watch the timing on the crosswalk, uh we can trim right on the beat."),
                (0, "Headlights streak through the downtown intersection as the camera tracks forward."),
                (1, "Keep that momentum carrying into the next cut."),
            ],
        ),
        Aurora => (
            &["Narrator", "Cinematographer"],
            &[
                (0, "High above the arctic ridge, ribbons of emerald aurora ripple across the sky."),
                (1, "The dynamic range in the highlights here, um gives us plenty of room in the grade."),
                (0, "Stars pierce through the curtain of green light above the snowline."),
                (1, "Let's preserve the deep shadow detail along the ridge."),
            ],
        ),
        Forest => (
            &["Narrator", "Cinematographer"],
            &[
                (0, "Sunlight filters through the old growth redwood canopy, carving shafts of warm mist."),
                (1, "Let's keep the natural ambience underneath the voiceover, um for this section."),
                (0, "Ferns catch the soft morning glow along the quiet trail."),
                (1, "That gentle pan gives us a clean transition point."),
            ],
        ),
        Dunes => (
            &["Narrator", "Director"],
            &[
                (0, "Out on the open dunes, the wind redraws every ridge line before evening."),
                (1, "That final camera drift is our closing shot, um let's lock the edit right there."),
                (0, "Long shadows stretch across the golden sand as the sun dips low."),
                (1, "Hold two more seconds on the horizon before fading out."),
            ],
        ),
        Plasma => (
            &["Sound Mixer", "Director"],
            &[
                (0, "Camera rolling, audio tone check one two, um scene one take one mark."),
                (1, "Playback looks clean and frame sync is locked across all tracks."),
                (0, "Levels are sitting right at reference with plenty of headroom."),
            ],
        ),
    };
    filmcraft_speech::acoustic::transcribe_scripted(audio, speakers, turns, opts, source_id)
}

fn transcribe_one_item(s: &Session, item: ItemId, t: &Arc<dyn Transcriber>, opts: &Options) -> Result<Option<Transcript>> {
    if s.transcriber.is_none()
        && let Some(ItemKind::Media(m)) = s.project.item(item).map(|i| &i.kind)
    {
        if !m.info.has_audio() {
            return Ok(None);
        }
        if let filmcraft_project::MediaRef::File { path } = &m.media
            && let Ok(Some(mut tr)) = filmcraft_speech::acoustic::transcribe_media_file(std::path::Path::new(path), models_dir().as_deref(), &t.id(), opts)
        {
            tr.normalize();
            return Ok(Some(tr));
        }
    }
    let Some(audio) = item_audio(s, item) else {
        return Ok(None);
    };
    if s.transcriber.is_none()
        && let Some(ItemKind::Media(m)) = s.project.item(item).map(|i| &i.kind)
        && let filmcraft_project::MediaRef::Generator(filmcraft_media::Generator::Demo(sc)) = m.media
    {
        // Instrumental score items (like Ambient_Score.wav on A2) have AudioOnly kind and no spoken dialogue.
        if m.info.kind == filmcraft_media::MediaKind::AudioOnly {
            let lang = opts.language.clone().unwrap_or_else(|| "en".into());
            return Ok(Some(Transcript { language: lang, source: t.id(), speakers: Vec::new(), words: Vec::new() }));
        }
        let mut tr = demo_scene_transcript(sc, &audio, opts, &t.id());
        tr.normalize();
        return Ok(Some(tr));
    }
    let mut tr = t.transcribe(&audio, opts, &mut |_, _| true).map_err(speech_err)?;
    tr.normalize();
    Ok(Some(tr))
}

/// Automatically transcribe any untranscribed audio clips in the active sequence when
/// `mediaAnalysis.autoTranscribe` is enabled.
pub fn maybe_auto_transcribe_sequence(s: &mut Session) {
    let ma = &s.prefs.media_analysis;
    if !ma.auto_transcribe || (ma.auto_transcribe_scope != "sequenceClips" && ma.auto_transcribe_scope != "allImported") {
        return;
    }
    let Some(q) = s.active_sequence() else { return };
    let raw: Vec<ItemId> = q.audio_tracks.iter().flat_map(|t| t.items.iter()).filter(|it| it.enabled).map(|it| it.item).collect();
    let mut missing = Vec::new();
    for i in raw {
        if let Some(m) = media_item(s, i)
            && !s.project.transcripts.contains_key(&m)
            && !missing.contains(&m)
        {
            missing.push(m);
        }
    }
    if missing.is_empty() {
        return;
    }
    let Ok(t) = transcriber(s, &Value::Null) else { return };
    let default_language = if ma.language_auto_detect { None } else { Some(ma.default_language.clone()) };
    let opts = Options { language: default_language, diarize: ma.speaker_labeling != "off", max_speakers: Options::default().max_speakers };
    let mut done = Vec::new();
    for item in missing {
        if let Ok(Some(tr)) = transcribe_one_item(s, item, &t, &opts) {
            done.push((item, tr));
        }
    }
    if !done.is_empty() {
        let pr = Arc::make_mut(&mut s.project);
        for (i, tr) in done {
            pr.transcripts.insert(i, Arc::new(tr));
        }
    }
}

/// Upgrade any real file item whose saved transcript was generated by the legacy `PHRASE_BANK`
/// fallback so opening or recovering an existing project replaces placeholder text with real
/// Whisper speech-to-text.
pub fn upgrade_legacy_file_transcripts(s: &mut Session) {
    if s.transcriber.is_some() {
        return;
    }
    let ma = &s.prefs.media_analysis;
    let default_language = if ma.language_auto_detect { None } else { Some(ma.default_language.clone()) };
    let opts = Options { language: default_language, diarize: ma.speaker_labeling != "off", max_speakers: Options::default().max_speakers };
    let stale: Vec<(ItemId, String)> = s
        .project
        .transcripts
        .iter()
        .filter_map(|(&item_id, tr)| {
            if !filmcraft_speech::acoustic::is_legacy_placeholder_transcript(tr) {
                return None;
            }
            if let Some(ItemKind::Media(m)) = s.project.item(item_id).map(|i| &i.kind)
                && let filmcraft_project::MediaRef::File { path } = &m.media
                && std::path::Path::new(path).exists()
            {
                Some((item_id, path.clone()))
            } else {
                None
            }
        })
        .collect();
    if stale.is_empty() {
        return;
    }
    let model_id = ma.whisper_model.clone();
    let mut upgraded = Vec::new();
    for (item_id, path) in stale {
        if let Ok(Some(mut tr)) = filmcraft_speech::acoustic::transcribe_media_file(std::path::Path::new(&path), models_dir().as_deref(), &model_id, &opts) {
            tr.normalize();
            upgraded.push((item_id, tr));
        }
    }
    if !upgraded.is_empty() {
        let pr = Arc::make_mut(&mut s.project);
        for (i, tr) in upgraded {
            pr.transcripts.insert(i, Arc::new(tr));
        }
    }
}

fn generate(s: &mut Session, p: &Value) -> Result<Value> {
    let items = targets(s, p);
    if items.is_empty() {
        return Err(bad("transcript.generate", "nothing to transcribe (pass `items`, select clips, or open a sequence with audio)"));
    }
    let t = transcriber(s, p)?;
    // Settings ▸ Media Analysis & Transcription: language (or auto-detect) and speaker labelling
    let ma = &s.prefs.media_analysis;
    let default_language = if ma.language_auto_detect { None } else { Some(ma.default_language.clone()) };
    let opts = Options {
        language: match str_p(p, "language") {
            Some(l) => Some(l).filter(|l| !l.is_empty() && *l != "auto").map(str::to_string),
            None => default_language,
        },
        diarize: bool_p(p, "diarize").unwrap_or(ma.speaker_labeling != "off"),
        max_speakers: u64_p(p, "maxSpeakers").map(|n| n.clamp(1, 32) as usize).unwrap_or(Options::default().max_speakers),
    };
    let mut done: Vec<(ItemId, Transcript)> = Vec::new();
    let mut skipped = Vec::new();
    for item in items {
        match transcribe_one_item(s, item, &t, &opts)? {
            Some(tr) => done.push((item, tr)),
            None => skipped.push(item.0),
        }
    }
    if done.is_empty() {
        return Err(EngineError::Other("none of the clips has audio to transcribe".into()));
    }
    let report: Vec<Value> = done
        .iter()
        .map(|(i, t)| json!({"item": i.0, "words": t.words.len(), "speakers": t.speakers.len(), "language": t.language, "source": t.source}))
        .collect();
    s.edit("Transcribe", move |pr, _| {
        for (i, t) in done {
            pr.transcripts.insert(i, Arc::new(t));
        }
        Ok(())
    })?;
    Ok(json!({"items": report, "skipped": skipped}))
}

fn set(s: &mut Session, p: &Value) -> Result<Value> {
    let item = u64_p(p, "item").map(ItemId).ok_or_else(|| bad("transcript.set", "`item` is required"))?;
    let item = media_item(s, item).ok_or_else(|| bad("transcript.set", "no such media item"))?;
    let v = p.get("transcript").cloned().ok_or_else(|| bad("transcript.set", "`transcript` is required"))?;
    let mut t: Transcript = serde_json::from_value(v).map_err(|e| bad("transcript.set", e.to_string()))?;
    if t.source.is_empty() {
        t.source = "imported".into();
    }
    t.normalize();
    t.check().map_err(|e| bad("transcript.set", e))?;
    let n = t.words.len();
    s.edit("Set Transcript", move |pr, _| {
        pr.transcripts.insert(item, Arc::new(t));
        Ok(())
    })?;
    Ok(json!({"item": item.0, "words": n}))
}

fn delete(s: &mut Session, p: &Value) -> Result<Value> {
    let items: Vec<ItemId> = match ids_p(p, "items").or_else(|| u64_p(p, "item").map(|i| vec![ItemId(i)])) {
        Some(v) => v.into_iter().filter_map(|i| media_item(s, i)).collect(),
        None => s.project.transcripts.keys().copied().collect(),
    };
    let n = items.iter().filter(|i| s.project.transcripts.contains_key(i)).count();
    if n == 0 {
        return Err(EngineError::Other("no transcript to delete".into()));
    }
    s.edit("Delete Transcript", move |pr, _| {
        for i in items {
            pr.transcripts.remove(&i);
        }
        Ok(())
    })?;
    Ok(json!({"deleted": n}))
}

fn word_json(i: usize, w: &SeqWord) -> Value {
    json!({"i": i, "text": w.text, "start": w.start.0, "end": w.end.0, "speaker": w.speaker, "clip": w.clip.0, "item": w.item.0, "confidence": w.confidence})
}

fn inspect(s: &mut Session, p: &Value) -> Result<Value> {
    let words = sequence_words(s);
    let gap = Tick::from_seconds_f64(f64_p(p, "paragraphGapSeconds").unwrap_or(1.5));
    let min_pause = Tick::from_seconds_f64(f64_p(p, "minPauseSeconds").unwrap_or(0.5));
    let paras: Vec<Value> = tx::paragraphs(&words, gap)
        .into_iter()
        .map(|r| {
            let text = words[r.clone()].iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
            json!({"from": r.start, "to": r.end - 1, "speaker": words[r.start].speaker, "start": words[r.start].start.0, "end": words[r.end - 1].end.0, "text": text})
        })
        .collect();
    let speakers: Vec<String> = {
        let mut v: Vec<String> = Vec::new();
        for w in &words {
            if let Some(n) = &w.speaker
                && !v.contains(n)
            {
                v.push(n.clone());
            }
        }
        v
    };
    let pauses: Vec<Value> = tx::sequence_pauses(&words, min_pause)
        .into_iter()
        .map(|ps| json!({"afterWord": ps.after_word, "start": ps.start.0, "end": ps.end.0, "seconds": ps.duration().seconds()}))
        .collect();
    let stats = tx::script_stats(&words, min_pause);
    let current = tx::word_at(&words, s.playhead());
    Ok(json!({
        "words": words.iter().enumerate().map(|(i, w)| word_json(i, w)).collect::<Vec<_>>(),
        "paragraphs": paras,
        "pauses": pauses,
        "speakers": speakers,
        "current": current,
        "stats": {
            "words": stats.word_count,
            "fillers": stats.filler_count,
            "pauses": stats.pause_count,
            "wpm": (stats.wpm * 10.0).round() / 10.0,
            "scenes": stats.scene_count,
        },
        "items": s.project.transcripts.iter().map(|(i, t)| json!({"item": i.0, "words": t.words.len(), "language": t.language, "source": t.source, "speakers": t.speakers.iter().map(|k| &k.name).collect::<Vec<_>>()})).collect::<Vec<_>>(),
    }))
}

fn search(s: &mut Session, p: &Value) -> Result<Value> {
    let q = str_p(p, "query").ok_or_else(|| bad("transcript.search", "`query` is required"))?;
    let words = sequence_words(s);
    let hits: Vec<Value> = tx::search(&words, q)
        .into_iter()
        .map(|r| json!({"from": r.start, "to": r.end - 1, "start": words[r.start].start.0, "end": words[r.end - 1].end.0}))
        .collect();
    Ok(json!({"matches": hits}))
}

/// Timeline range of the words `from..=to` (frame-snapped outward).
fn range_p(s: &Session, p: &Value, cmd: &str) -> Result<TimeRange> {
    let words = sequence_words(s);
    let from = u64_p(p, "from").ok_or_else(|| bad(cmd, "`from` (word index) is required"))? as usize;
    let to = u64_p(p, "to").map(|n| n as usize).unwrap_or(from);
    tx::word_range(&words, from, to, s.sequence_rate()).ok_or_else(|| bad(cmd, format!("word index out of range (the transcript has {} words)", words.len())))
}

fn range_json(r: TimeRange) -> Value {
    json!({"start": r.start.0, "end": r.end().0})
}

fn select(s: &mut Session, p: &Value) -> Result<Value> {
    let r = range_p(s, p, "transcript.select")?;
    let fd = s.sequence_rate().frame_duration();
    s.edit_sequence("Mark Transcript Selection", |q, _, _| {
        q.mark_in = Some(r.start);
        q.mark_out = Some(r.end() - fd);
        Ok(())
    })?;
    s.set_playhead(r.start);
    Ok(range_json(r))
}

fn extract_or_lift(s: &mut Session, p: &Value, extract: bool) -> Result<Value> {
    let cmd = if extract { "transcript.extract" } else { "transcript.lift" };
    let r = range_p(s, p, cmd)?;
    let tg = s.targeting().targeted;
    s.edit_sequence(if extract { "Extract Text" } else { "Lift Text" }, |q, ctx, _| {
        if extract {
            edit::extract(q, &tg, r, ctx);
        } else {
            edit::lift(q, &tg, r, ctx);
        }
        q.mark_in = None;
        q.mark_out = None;
        Ok(())
    })?;
    s.set_playhead(r.start);
    Ok(range_json(r))
}

fn rename_speaker(s: &mut Session, p: &Value) -> Result<Value> {
    let name = str_p(p, "name").map(str::trim).filter(|n| !n.is_empty()).ok_or_else(|| bad("transcript.renameSpeaker", "`name` is required"))?.to_string();
    let item = u64_p(p, "item").map(ItemId).and_then(|i| media_item(s, i));
    // `speaker`: the current name (every transcript), or an index (needs `item`)
    let (old_name, index) = match p.get("speaker") {
        Some(Value::String(n)) => (Some(n.clone()), None),
        Some(v) if v.is_u64() => (None, v.as_u64().map(|n| n as usize)),
        _ => return Err(bad("transcript.renameSpeaker", "`speaker` (name, or index with `item`) is required")),
    };
    if index.is_some() && item.is_none() {
        return Err(bad("transcript.renameSpeaker", "a speaker index needs `item`"));
    }
    let mut n = 0;
    let mut next = s.project.transcripts.clone();
    for (i, t) in next.iter_mut() {
        if item.is_some_and(|x| x != *i) {
            continue;
        }
        let tt = Arc::make_mut(t);
        for (k, sp) in tt.speakers.iter_mut().enumerate() {
            if old_name.as_ref().is_some_and(|o| *o == sp.name) || index == Some(k) {
                sp.name = name.clone();
                n += 1;
            }
        }
    }
    if n == 0 {
        return Err(EngineError::Other("no such speaker".into()));
    }
    s.edit("Rename Speaker", move |pr, _| {
        pr.transcripts = next;
        Ok(())
    })?;
    Ok(json!({"renamed": n}))
}

/// Correct the text and/or speaker of a word in the sequence transcript (`word` index) or in a
/// media clip's transcript (`item` + `itemWord`). If `text` contains multiple whitespace-separated
/// words, the original word's media time interval is split proportionally across the new words.
fn edit_word(s: &mut Session, p: &Value) -> Result<Value> {
    let (target_item, item_wi) = if let Some(wi) = u64_p(p, "word").map(|n| n as usize) {
        let words = sequence_words(s);
        let sw = words.get(wi).ok_or_else(|| bad("transcript.editWord", format!("word index {wi} out of range ({})", words.len())))?;
        (sw.item, sw.index)
    } else {
        let it =
            u64_p(p, "item").map(ItemId).and_then(|i| media_item(s, i)).ok_or_else(|| bad("transcript.editWord", "need `word` or (`item` and `itemWord`)"))?;
        let iwi = u64_p(p, "itemWord").ok_or_else(|| bad("transcript.editWord", "`itemWord` is required when `item` is used"))? as usize;
        (it, iwi)
    };

    let new_text = str_p(p, "text").map(|t| t.trim().to_string());
    let speaker_arg = p.get("speaker").cloned();
    if new_text.is_none() && speaker_arg.is_none() {
        return Err(bad("transcript.editWord", "pass `text` and/or `speaker`"));
    }

    let mut next = s.project.transcripts.clone();
    let tr_arc = next.get_mut(&target_item).ok_or_else(|| bad("transcript.editWord", "item has no transcript"))?;
    let tr = Arc::make_mut(tr_arc);
    if item_wi >= tr.words.len() {
        return Err(bad("transcript.editWord", "item word index out of range"));
    }

    let resolved_speaker: Option<u32> = match &speaker_arg {
        Some(Value::Number(n)) => n.as_u64().map(|v| v as u32),
        Some(Value::String(name)) => {
            let trimmed = name.trim();
            if trimmed.is_empty() {
                None
            } else if let Some(pos) = tr.speakers.iter().position(|sp| sp.name.eq_ignore_ascii_case(trimmed)) {
                Some(pos as u32)
            } else {
                tr.speakers.push(filmcraft_project::Speaker { name: trimmed.to_string() });
                Some((tr.speakers.len() - 1) as u32)
            }
        }
        _ => None,
    };

    if let Some(sp) = resolved_speaker {
        // Optional range support: `toWord` in sequence words or single word
        if let (Some(from_w), Some(to_w)) = (u64_p(p, "word").map(|n| n as usize), u64_p(p, "toWord").map(|n| n as usize)) {
            let words = sequence_words(s);
            let (lo, hi) = (from_w.min(to_w), from_w.max(to_w));
            for sw in words.iter().take(hi + 1).skip(lo) {
                if let Some(t_arc) = next.get_mut(&sw.item) {
                    let tt = Arc::make_mut(t_arc);
                    let sp_idx = if let Some(Value::String(name)) = &speaker_arg {
                        let trimmed = name.trim();
                        if let Some(pos) = tt.speakers.iter().position(|s| s.name.eq_ignore_ascii_case(trimmed)) {
                            pos as u32
                        } else {
                            tt.speakers.push(filmcraft_project::Speaker { name: trimmed.to_string() });
                            (tt.speakers.len() - 1) as u32
                        }
                    } else {
                        sp
                    };
                    if let Some(w) = tt.words.get_mut(sw.index) {
                        w.speaker = Some(sp_idx);
                    }
                    tt.normalize();
                }
            }
        } else {
            let tr = Arc::make_mut(next.get_mut(&target_item).ok_or_else(|| bad("transcript.editWord", "no transcript"))?);
            tr.words[item_wi].speaker = Some(sp);
        }
    }

    if let Some(txt) = new_text {
        let tr = Arc::make_mut(next.get_mut(&target_item).ok_or_else(|| bad("transcript.editWord", "no transcript"))?);
        let tokens: Vec<&str> = txt.split_whitespace().filter(|w| !w.is_empty()).collect();
        if tokens.is_empty() {
            tr.words.remove(item_wi);
        } else if tokens.len() == 1 {
            tr.words[item_wi].text = tokens[0].to_string();
        } else {
            let orig = tr.words.remove(item_wi);
            let span = (orig.end - orig.start).max(Tick(tokens.len() as i64));
            let n = tokens.len() as i64;
            for (k, tok) in tokens.into_iter().enumerate() {
                let s0 = orig.start + Tick(span.0 * k as i64 / n);
                let e0 = orig.start + Tick(span.0 * (k as i64 + 1) / n);
                let mut nw = filmcraft_project::Word::new(tok, s0, e0.max(s0));
                nw.speaker = orig.speaker;
                nw.confidence = orig.confidence;
                tr.words.insert(item_wi + k, nw);
            }
        }
        tr.normalize();
    } else if let Some(t_arc) = next.get_mut(&target_item) {
        Arc::make_mut(t_arc).normalize();
    }

    s.edit("Correct Transcript Text", move |pr, _| {
        pr.transcripts = next;
        Ok(())
    })?;
    Ok(json!({"item": target_item.0, "word": item_wi}))
}

/// Find and replace words or phrases across the active sequence's transcripts (or a specific item).
fn replace_text(s: &mut Session, p: &Value) -> Result<Value> {
    let find = str_p(p, "find").map(str::trim).filter(|q| !q.is_empty()).ok_or_else(|| bad("transcript.replace", "`find` is required"))?;
    let rep = str_p(p, "replace").ok_or_else(|| bad("transcript.replace", "`replace` is required"))?.trim();
    let only_item = u64_p(p, "item").map(ItemId).and_then(|i| media_item(s, i));
    let max_replacements = u64_p(p, "limit").map(|n| n as usize).unwrap_or(usize::MAX);

    let words = sequence_words(s);
    let matches = tx::search(&words, find);
    if matches.is_empty() {
        return Ok(json!({"replaced": 0}));
    }

    let mut next = s.project.transcripts.clone();
    let mut replaced = 0usize;
    // Process in reverse order so word indices inside each media transcript stay valid
    for m in matches.into_iter().rev() {
        if replaced >= max_replacements || m.is_empty() {
            continue;
        }
        let first_sw = &words[m.start];
        if only_item.is_some_and(|it| it != first_sw.item) {
            continue;
        }
        // Ensure all words in the phrase belong to the same media item and are contiguous
        let same_item = words[m.clone()].iter().enumerate().all(|(k, w)| w.item == first_sw.item && w.index == first_sw.index + k);
        if !same_item {
            continue;
        }
        let Some(tr_arc) = next.get_mut(&first_sw.item) else { continue };
        let tr = Arc::make_mut(tr_arc);
        let start_idx = first_sw.index;
        let end_idx = start_idx + m.len();
        if end_idx > tr.words.len() {
            continue;
        }
        let t_start = tr.words[start_idx].start;
        let t_end = tr.words[end_idx - 1].end;
        let sp = tr.words[start_idx].speaker;
        let conf = tr.words[start_idx].confidence;
        tr.words.drain(start_idx..end_idx);

        let tokens: Vec<&str> = rep.split_whitespace().filter(|w| !w.is_empty()).collect();
        if !tokens.is_empty() {
            let span = (t_end - t_start).max(Tick(tokens.len() as i64));
            let n = tokens.len() as i64;
            for (k, tok) in tokens.into_iter().enumerate() {
                let s0 = t_start + Tick(span.0 * k as i64 / n);
                let e0 = t_start + Tick(span.0 * (k as i64 + 1) / n);
                let mut nw = filmcraft_project::Word::new(tok, s0, e0.max(s0));
                nw.speaker = sp;
                nw.confidence = conf;
                tr.words.insert(start_idx + k, nw);
            }
        }
        tr.normalize();
        replaced += 1;
    }

    if replaced > 0 {
        s.edit("Replace Transcript Text", move |pr, _| {
            pr.transcripts = next;
            Ok(())
        })?;
    }
    Ok(json!({"replaced": replaced}))
}

/// Split the sequence clips at a word's frame-aligned start (or end when `after` is true), creating
/// a Descript-style scene boundary (`/`).
fn split_at_word(s: &mut Session, p: &Value) -> Result<Value> {
    let words = sequence_words(s);
    let wi = u64_p(p, "word").ok_or_else(|| bad("transcript.splitAtWord", "`word` index is required"))? as usize;
    let after = bool_p(p, "after").unwrap_or(false);
    let w = words.get(wi).ok_or_else(|| bad("transcript.splitAtWord", format!("word index {wi} out of range")))?;
    let rate = s.sequence_rate();
    let raw_t = if after { w.end } else { w.start };
    let mut cut = rate.snap(raw_t);
    if after && cut < raw_t {
        cut += rate.frame_duration();
    }
    let split_ids = s.edit_sequence("Split at Transcript Word", |q, ctx, _| Ok(edit::razor(q, &[], cut, ctx)))?;
    s.set_playhead(cut);
    Ok(json!({"time": cut.0, "clips": split_ids.into_iter().map(|c| c.0).collect::<Vec<_>>()}))
}

/// Ripple-delete a single pause immediately after `afterWord`, keeping `keepSeconds` of room tone.
fn delete_pause(s: &mut Session, p: &Value) -> Result<Value> {
    let words = sequence_words(s);
    let after_word = u64_p(p, "afterWord").ok_or_else(|| bad("transcript.deletePause", "`afterWord` is required"))? as usize;
    let keep = Tick::from_seconds_f64(f64_p(p, "keepSeconds").unwrap_or(0.05));
    let Some(r) = tx::single_pause_range(&words, after_word, keep, s.sequence_rate()) else {
        return Ok(json!({"removed": 0, "ticks": 0}));
    };
    remove_ranges(s, "Delete Pause", vec![r])
}

/// Insert or overwrite a word range `[from..=to]` from a source media clip's transcript directly
/// into the active sequence at the playhead.
fn place_from_source(s: &mut Session, p: &Value, overwrite: bool) -> Result<Value> {
    let cmd = if overwrite { "transcript.overwriteFromSource" } else { "transcript.insertFromSource" };
    let raw_item = u64_p(p, "item")
        .map(ItemId)
        .or(s.state.source_item)
        .or_else(|| s.state.project_selection.first().copied())
        .ok_or_else(|| bad(cmd, "pass `item` or open a clip in the Source Monitor"))?;
    let item = media_item(s, raw_item).ok_or_else(|| bad(cmd, "not a media item"))?;
    let tr = s.project.transcripts.get(&item).ok_or_else(|| bad(cmd, "source clip has no transcript"))?.clone();
    let from = u64_p(p, "from").ok_or_else(|| bad(cmd, "`from` word index is required"))? as usize;
    let to = u64_p(p, "to").map(|n| n as usize).unwrap_or(from);
    let rate = s.project.item(item).map(|i| i.frame_rate()).unwrap_or_else(|| s.sequence_rate());
    let range = tx::media_word_range(&tr, from, to, rate).ok_or_else(|| bad(cmd, "source word index out of range"))?;
    let fd = rate.frame_duration();

    s.edit(if overwrite { "Overwrite from Script" } else { "Insert from Script" }, move |pr, st| {
        st.source_item = Some(item);
        if let Some(pi) = pr.item_mut(item)
            && let ItemKind::Media(m) = &mut pi.kind
        {
            m.mark_in = Some(range.start);
            m.mark_out = Some((range.end() - fd).max(range.start));
        }
        Ok(())
    })?;
    let edit_cmd = if overwrite { "sequence.overwrite" } else { "sequence.insert" };
    s.execute(edit_cmd, json!({}))
}

/// Export the sequence or media transcript as Descript-style Markdown, plain text, or JSON.
fn export_transcript(s: &mut Session, p: &Value) -> Result<Value> {
    let fmt = str_p(p, "format").unwrap_or("markdown").to_ascii_lowercase();
    let gap = Tick::from_seconds_f64(f64_p(p, "paragraphGapSeconds").unwrap_or(1.5));
    let min_pause = Tick::from_seconds_f64(f64_p(p, "minPauseSeconds").unwrap_or(0.5));
    let words = if let Some(item_id) = u64_p(p, "item").map(ItemId).and_then(|i| media_item(s, i)) {
        let tr = s.project.transcripts.get(&item_id).ok_or_else(|| bad("transcript.export", "item has no transcript"))?;
        tr.words
            .iter()
            .enumerate()
            .map(|(i, w)| SeqWord {
                text: w.text.clone(),
                start: w.start,
                end: w.end,
                clip: filmcraft_project::ClipId(0),
                item: item_id,
                index: i,
                track: 0,
                speaker: tr.speaker_name(w),
                confidence: w.confidence,
            })
            .collect()
    } else {
        sequence_words(s)
    };
    let title = s.state.active_sequence.and_then(|id| s.project.item(id)).map(|i| i.name.clone()).unwrap_or_else(|| "Sequence Script".into());
    let rate = s.sequence_rate();
    let df = s.active_sequence().is_some_and(|q| q.settings.drop_frame);

    let text = match fmt.as_str() {
        "text" | "txt" => tx::format_script_text(&words, gap),
        "json" => serde_json::to_string_pretty(&words.iter().enumerate().map(|(i, w)| word_json(i, w)).collect::<Vec<_>>()).unwrap_or_default(),
        _ => tx::format_script_markdown(&title, &words, gap, min_pause, rate, df),
    };
    if let Some(path) = str_p(p, "path").filter(|path| !path.is_empty()) {
        s.services.write_file(path, text.as_bytes()).map_err(|e| EngineError::Other(e.to_string()))?;
    }
    Ok(json!({"format": fmt, "text": text, "words": words.len()}))
}

fn remove_ranges(s: &mut Session, label: &str, ranges: Vec<TimeRange>) -> Result<Value> {
    let n = ranges.len();
    if n == 0 {
        return Ok(json!({"removed": 0, "ticks": 0}));
    }
    let total = s.edit_sequence(label, |q, ctx, _| Ok(tx::ripple_delete_ranges(q, ranges, ctx)))?;
    Ok(json!({"removed": n, "ticks": total.0, "seconds": total.0 as f64 / TICKS_PER_SECOND as f64}))
}

fn remove_pauses(s: &mut Session, p: &Value) -> Result<Value> {
    let words = sequence_words(s);
    let min = Tick::from_seconds_f64(f64_p(p, "minSeconds").unwrap_or(1.0));
    let keep = Tick::from_seconds_f64(f64_p(p, "keepSeconds").unwrap_or(0.15));
    let ranges = tx::find_pauses(&words, min, keep, s.sequence_rate());
    remove_ranges(s, "Remove Pauses", ranges)
}

fn remove_fillers(s: &mut Session, p: &Value) -> Result<Value> {
    let words = sequence_words(s);
    let fillers: Vec<String> = match p.get("fillers").and_then(Value::as_array) {
        Some(a) => a.iter().filter_map(Value::as_str).map(str::to_string).collect(),
        None => tx::DEFAULT_FILLERS.iter().map(|f| f.to_string()).collect(),
    };
    let hits = tx::find_fillers(&words, &fillers);
    let ranges = tx::filler_ranges(&words, &hits, s.sequence_rate());
    remove_ranges(s, "Remove Filler Words", ranges)
}

fn create_captions(s: &mut Session, p: &Value) -> Result<Value> {
    let words = sequence_words(s);
    let d = CaptionRules::default();
    let rules = CaptionRules {
        max_chars: u64_p(p, "maxChars").map(|n| n as usize).unwrap_or(d.max_chars),
        lines: u64_p(p, "lines").map(|n| n as usize).unwrap_or(d.lines),
        min_duration: f64_p(p, "minSeconds").map(Tick::from_seconds_f64).unwrap_or(d.min_duration),
        max_duration: f64_p(p, "maxSeconds").map(Tick::from_seconds_f64).unwrap_or(d.max_duration),
        gap_frames: p.get("gapFrames").and_then(Value::as_i64).unwrap_or(d.gap_frames),
        break_pause: d.break_pause,
    };
    let blocks = tx::caption_blocks(&words, &rules, s.sequence_rate());
    let format = str_p(p, "format").and_then(CaptionFormat::from_name).unwrap_or_default();
    let name = str_p(p, "name").unwrap_or("Transcript").to_string();
    let n = blocks.len();
    let tid = s.edit_sequence("Create Captions", |q, ctx, st| {
        let tid = TrackId(ctx.alloc());
        let mut t = CaptionTrack::new(tid, name, format);
        t.captions = tx::blocks_to_captions(&blocks, ctx);
        q.caption_tracks.insert(0, t);
        st.caption_selection.clear();
        Ok(tid)
    })?;
    Ok(json!({"track": tid.0, "captions": n}))
}

fn models(_: &mut Session, _: &Value) -> Result<Value> {
    let dir = models_dir();
    Ok(json!({
        "available": filmcraft_speech::available(),
        "whisperAvailable": filmcraft_speech::whisper_available(),
        "default": filmcraft_speech::models::DEFAULT_MODEL,
        "dir": dir.as_ref().map(|d| d.to_string_lossy().to_string()),
        "models": filmcraft_speech::models::catalogue().iter().map(|m| json!({
            "id": m.id, "name": m.name, "multilingual": m.multilingual, "description": m.description,
            "license": m.license, "source": m.source, "size": m.size(),
            "installed": dir.as_ref().is_some_and(|d| filmcraft_speech::models::installed(d, m)),
        })).collect::<Vec<_>>(),
    }))
}

/// Download a catalogue model into `<data dir>/models` (feature `speech-download`). Hosts show
/// the size, source and licence (`transcript.models`) and ask before running this.
fn download_model(_: &mut Session, p: &Value) -> Result<Value> {
    let id = str_p(p, "model").unwrap_or(filmcraft_speech::models::DEFAULT_MODEL);
    let m = filmcraft_speech::models::find(id).ok_or_else(|| speech_err(SpeechError::UnknownModel(id.into())))?;
    let dir = models_dir().ok_or_else(|| EngineError::Other("no data directory for speech models".into()))?;
    #[cfg(feature = "speech-download")]
    {
        filmcraft_speech::models::download(&dir, m, &mut |_, _, _| true).map_err(speech_err)?;
        Ok(json!({"model": m.id, "dir": filmcraft_speech::models::model_dir(&dir, m).to_string_lossy()}))
    }
    #[cfg(not(feature = "speech-download"))]
    {
        let _ = (m, dir);
        Err(EngineError::Other("model downloads are not available in this build (built without the `speech-download` feature)".into()))
    }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            "transcript.generate",
            "Transcribe…",
            &["Sequence", "Transcript"],
            r#"{"items":[id]?,"model":"whisper-base"?,"language":"en|auto"?,"diarize":bool?,"maxSpeakers":n?}"#,
            can_transcribe,
            generate,
            true,
        ),
        spec(
            "transcript.set",
            "Import Transcript",
            &[],
            r#"{"item":id,"transcript":{"language":str,"speakers":[{"name":str}],"words":[{"text":str,"start":tick,"end":tick,"speaker":n?}]}}"#,
            always,
            set,
            true,
        ),
        spec("transcript.delete", "Delete Transcript", &["Sequence", "Transcript"], r#"{"items":[id]?}"#, has_transcripts, delete, true),
        spec("transcript.inspect", "Inspect Transcript", &[], r#"{"paragraphGapSeconds":f?,"minPauseSeconds":f?}"#, always, inspect, false),
        spec("transcript.search", "Search Transcript", &[], r#"{"query":str}"#, always, search, false),
        spec("transcript.models", "List Speech Models", &[], "{}", always, models, false),
        spec("transcript.downloadModel", "Download Speech Model", &[], r#"{"model":"whisper-base"?}"#, can_download, download_model, true),
        spec("transcript.select", "Mark Selected Text", &[], r#"{"from":word,"to":word?}"#, has_transcript, select, true),
        spec("transcript.extract", "Extract Selected Text", &[], r#"{"from":word,"to":word?}"#, has_transcript, |s, p| extract_or_lift(s, p, true), true),
        spec("transcript.lift", "Lift Selected Text", &[], r#"{"from":word,"to":word?}"#, has_transcript, |s, p| extract_or_lift(s, p, false), true),
        spec(
            "transcript.renameSpeaker",
            "Rename Speaker…",
            &[],
            r#"{"speaker":"Speaker 1"|index,"name":str,"item":id?}"#,
            has_transcripts,
            rename_speaker,
            true,
        ),
        spec(
            "transcript.editWord",
            "Correct Transcript Word",
            &[],
            r#"{"word":index?,"toWord":index?,"item":id?,"itemWord":index?,"text":str?,"speaker":str|index?}"#,
            has_transcripts,
            edit_word,
            true,
        ),
        spec(
            "transcript.replace",
            "Find and Replace in Transcript",
            &["Sequence", "Transcript"],
            r#"{"find":str,"replace":str,"item":id?,"limit":n?}"#,
            has_transcript,
            replace_text,
            true,
        ),
        spec(
            "transcript.splitAtWord",
            "Split Clip at Word",
            &["Sequence", "Transcript"],
            r#"{"word":index,"after":bool?}"#,
            has_transcript,
            split_at_word,
            true,
        ),
        spec("transcript.deletePause", "Delete Pause", &[], r#"{"afterWord":index,"keepSeconds":f?}"#, has_transcript, delete_pause, true),
        spec(
            "transcript.insertFromSource",
            "Insert Selected Source Words",
            &[],
            r#"{"item":id?,"from":word,"to":word?}"#,
            has_seq,
            |s, p| place_from_source(s, p, false),
            true,
        ),
        spec(
            "transcript.overwriteFromSource",
            "Overwrite Selected Source Words",
            &[],
            r#"{"item":id?,"from":word,"to":word?}"#,
            has_seq,
            |s, p| place_from_source(s, p, true),
            true,
        ),
        spec(
            "transcript.export",
            "Export Transcript Script",
            &["Sequence", "Transcript"],
            r#"{"format":"markdown|text|json"?,"path":str?,"item":id?}"#,
            has_transcripts,
            export_transcript,
            false,
        ),
        spec(
            "transcript.removePauses",
            "Remove Pauses",
            &["Sequence", "Transcript"],
            r#"{"minSeconds":f?,"keepSeconds":f?}"#,
            has_transcript,
            remove_pauses,
            true,
        ),
        spec("transcript.removeFillers", "Remove Filler Words", &["Sequence", "Transcript"], r#"{"fillers":[str]?}"#, has_transcript, remove_fillers, true),
        spec(
            "transcript.createCaptions",
            "Create Captions from Transcript…",
            &["Sequence", "Transcript"],
            r#"{"maxChars":n?,"lines":1|2?,"minSeconds":f?,"maxSeconds":f?,"gapFrames":n?,"format":str?,"name":str?}"#,
            has_transcript,
            create_captions,
            true,
        ),
    ]
}
