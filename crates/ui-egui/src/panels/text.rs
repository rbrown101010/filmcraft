//! The Text panel: Transcript (Descript-style Script Editor + Premiere Text-Based Editing) /
//! Captions / Graphics tabs.
//!
//! - **Transcript tab**: Full Descript-style script editor integrated with Premiere Pro's
//!   text-based editing. Supports Sequence Script and Source Clip Script views, `Cut Media` vs
//!   `Correct Text` modes, click / Shift+click / click-and-drag word range selection, inline
//!   double-click word correction (`transcript.editWord`), Find & Replace (`transcript.replace`),
//!   Scene `/` boundary dividers & splitting (`transcript.splitAtWord`), inline clickable pause
//!   pills (`[0.8s]`, `transcript.deletePause`), filler word amber highlights & bulk removal
//!   (`transcript.removeFillers`), pause shortening (`transcript.removePauses`), speaker color
//!   badges with inline speaker rename (`transcript.renameSpeaker`), source-to-sequence phrase
//!   insert/overwrite (`transcript.insertFromSource`, `transcript.overwriteFromSource`), script
//!   export (`transcript.export`), and 1-click Auto-Transcribe (`mediaAnalysis.autoTranscribe`).
//! - **Captions tab**: Lists caption segments of a caption track with editable In/Out timecodes
//!   and text, toolbar (add, split, merge, delete, export), track picker, and style strip.
//! - **Graphics tab**: Lists all graphic text layers in the active sequence in timeline order with
//!   live search, bulk find-and-replace, inline text editing (`graphics.setText`), and playhead
//!   jump.

use egui::{Align2, Color32, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use filmcraft_edit::transcript as tx;
use filmcraft_project::graphic::{LayerContent, eval_layer, layer_indices};
use filmcraft_project::{CaptionAlign, CaptionAnchor, CaptionFormat, ClipId, ItemId, ItemKind};
use filmcraft_time::{Tick, TimeDisplay, format_time};
use serde_json::{Value, json};

use crate::FilmcraftApp;
use crate::icons::{self, Icon};
use crate::theme::Tokens;

const TABS: [&str; 3] = ["Transcript", "Captions", "Graphics"];

const SPEAKER_COLORS: [Color32; 6] = [
    Color32::from_rgb(78, 168, 246),  // Sky Blue
    Color32::from_rgb(174, 122, 242), // Iris Purple
    Color32::from_rgb(82, 196, 152),  // Emerald Teal
    Color32::from_rgb(240, 152, 72),  // Mango Amber
    Color32::from_rgb(232, 108, 146), // Rose Pink
    Color32::from_rgb(218, 196, 78),  // Gold
];

fn speaker_color(speaker: Option<&str>) -> Color32 {
    let Some(s) = speaker else {
        return SPEAKER_COLORS[0];
    };
    let h = s.bytes().fold(0usize, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize));
    SPEAKER_COLORS[h % SPEAKER_COLORS.len()]
}

#[derive(Clone, Debug)]
struct WordEditState {
    word_idx: usize,
    item: Option<u64>,
    item_word: usize,
    buf: String,
}

#[derive(Clone, Debug)]
struct SpeakerEditState {
    old_name: String,
    buf: String,
}

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    ui.painter().rect_filled(rect, 0.0, t.panel_bg);
    // tabs
    let mut x = rect.min.x + 12.0;
    for tab in TABS {
        let w = tab.len() as f32 * 7.0 + 16.0;
        let r = Rect::from_min_size(pos2(x, rect.min.y + 4.0), vec2(w, 24.0));
        let resp = ui.interact(r, egui::Id::new(("text-tab", tab)), Sense::click());
        let active = app.ui.text_tab == tab;
        ui.painter().text(
            pos2(r.min.x, r.center().y),
            Align2::LEFT_CENTER,
            tab,
            if active { Tokens::semibold(12.5) } else { Tokens::ui(12.5) },
            if active { t.text } else { t.text_dim },
        );
        if active {
            ui.painter().line_segment([pos2(r.min.x, r.max.y), pos2(r.min.x + w - 16.0, r.max.y)], Stroke::new(2.0, t.text));
        }
        app.auto.add(&format!("text.tab.{tab}"), r, tab);
        if resp.clicked() {
            app.ui.text_tab = tab.to_string();
        }
        x += w + 6.0;
    }
    let body = Rect::from_min_max(pos2(rect.min.x, rect.min.y + 34.0), rect.max);
    match app.ui.text_tab.as_str() {
        "Captions" => captions(app, ui, body),
        "Transcript" => transcript(app, ui, body),
        _ => graphics_tab(app, ui, body),
    }
}

fn tool_button(app: &mut FilmcraftApp, ui: &mut egui::Ui, r: Rect, icon: Icon, id: &str, label: &str, enabled: bool) -> bool {
    let t = app.tokens;
    let resp = ui.interact(r, egui::Id::new(("text-tool", id)), if enabled { Sense::click() } else { Sense::hover() });
    if enabled && resp.hovered() {
        ui.painter().rect_filled(r, 3.0, t.hover);
    }
    icons::paint(ui.painter(), r.shrink(5.0), icon, if enabled { t.icon } else { t.text_faint });
    app.auto.add(id, r, label);
    resp.on_hover_text(label).clicked() && enabled
}

fn chip_button(app: &mut FilmcraftApp, ui: &mut egui::Ui, r: Rect, id: &str, text: &str, active: bool, tooltip: &str) -> bool {
    let t = app.tokens;
    let resp = ui.interact(r, egui::Id::new(("text-chip", id)), Sense::click());
    let bg = if active {
        t.accent
    } else if resp.hovered() {
        t.hover
    } else {
        t.field_bg
    };
    ui.painter().rect_filled(r, 4.0, bg);
    let fg = if active { Color32::WHITE } else { t.text };
    ui.painter().text(r.center(), Align2::CENTER_CENTER, text, Tokens::ui(11.0), fg);
    app.auto.add(id, r, tooltip);
    resp.on_hover_text(tooltip).clicked()
}

fn captions(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    let Some(seq) = app.session.active_sequence().cloned() else {
        crate::dock::placeholder(ui, rect, &t, "Open a sequence to work with captions");
        return;
    };
    let mut actions: Vec<(String, Value)> = Vec::new();
    if seq.caption_tracks.is_empty() {
        let c = rect.center();
        icons::paint(ui.painter(), Rect::from_center_size(c - vec2(0.0, 70.0), vec2(40.0, 40.0)), Icon::Captions, t.text_dim);
        ui.painter().text(c - vec2(0.0, 30.0), Align2::CENTER_CENTER, "Add captions", Tokens::semibold(16.0), t.text);
        ui.painter().text(c - vec2(0.0, 8.0), Align2::CENTER_CENTER, "Create a caption track or import a caption file.", Tokens::ui(12.0), t.text_dim);
        for (i, (id, label, cmd)) in
            [("text.captions.newTrack", "Create new caption track", "captions.newTrack"), ("text.captions.import", "Import captions file…", "captions.import")]
                .into_iter()
                .enumerate()
        {
            let r = Rect::from_center_size(c + vec2(0.0, 26.0 + i as f32 * 34.0), vec2(200.0, 26.0));
            let resp = ui.interact(r, egui::Id::new(id), Sense::click());
            ui.painter().rect_filled(r, 13.0, if i == 0 { if resp.hovered() { t.accent_hover } else { t.accent } } else { t.field_bg });
            ui.painter().text(r.center(), Align2::CENTER_CENTER, label, Tokens::semibold(12.0), Color32::WHITE);
            app.auto.add(id, r, label);
            if resp.clicked() {
                actions.push((cmd.into(), json!({})));
            }
        }
        run(app, ui, actions);
        return;
    }
    let rate = seq.settings.frame_rate;
    let df = seq.settings.drop_frame;
    let tc = |x: filmcraft_time::Tick| format_time(x, rate, df, TimeDisplay::Timecode, 48_000);
    // the track shown: the one holding the first selected caption, else C1
    let sel = app.session.state.caption_selection.clone();
    let track_idx = sel.first().and_then(|c| seq.caption_tracks.iter().position(|tr| tr.caption(*c).is_some())).unwrap_or(0);
    let track_idx = ui.ctx().data(|d| d.get_temp::<usize>(egui::Id::new("text-cap-track"))).filter(|i| *i < seq.caption_tracks.len()).unwrap_or(track_idx);
    let track = &seq.caption_tracks[track_idx];

    // ---- toolbar: search, track picker, add / split / merge / delete
    let bar = Rect::from_min_size(rect.min + vec2(10.0, 2.0), vec2(rect.width() - 20.0, 26.0));
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(Rect::from_min_size(bar.min, vec2(170.0f32.min(bar.width() * 0.4), 24.0))));
    let sresp = crate::widgets::search_field(&mut child, &mut app.ui.caption_search, "Search", 170.0f32.min(bar.width() * 0.4), &t);
    app.auto.add("text.captions.search", sresp.rect, "Search captions");
    let mut x = bar.min.x + 180.0f32.min(bar.width() * 0.4 + 10.0);
    let picker = Rect::from_min_size(pos2(x, bar.min.y + 1.0), vec2(130.0, 22.0));
    let label = format!("C{} · {}", track_idx + 1, track.name);
    let presp = crate::widgets::dropdown_text(ui, picker, &label, &t, egui::Id::new("text-cap-track-picker"));
    app.auto.add("text.captions.track", picker, "Caption track");
    egui::Popup::menu(&presp).show(|ui| {
        for (i, tr) in seq.caption_tracks.iter().enumerate() {
            if ui.selectable_label(i == track_idx, format!("C{} · {} ({})", i + 1, tr.name, tr.format.label())).clicked() {
                ui.ctx().data_mut(|d| d.insert_temp(egui::Id::new("text-cap-track"), i));
            }
        }
        ui.separator();
        for f in CaptionFormat::ALL {
            if ui.button(format!("New {} track", f.label())).clicked() {
                actions.push(("captions.newTrack".into(), json!({"format": f.label()})));
            }
        }
    });
    x = picker.max.x + 8.0;
    let ph = app.session.playhead();
    let any_sel = !sel.is_empty();
    let under = track.caption_at(ph).filter(|c| c.start < ph).map(|c| c.id);
    let tools: [(Icon, &str, &str, bool, &str, Value); 5] = [
        (Icon::Plus, "text.captions.add", "Add caption at playhead", track.caption_at(ph).is_none(), "captions.add", json!({"track": track.id.0})),
        (
            Icon::Razor,
            "text.captions.split",
            "Split caption at playhead",
            under.is_some(),
            "captions.split",
            json!({"caption": under.map(|c| c.0), "time": ph.0}),
        ),
        (Icon::Link, "text.captions.merge", "Merge selected captions", sel.len() > 1, "captions.merge", json!({})),
        (Icon::Trash, "text.captions.delete", "Delete selected captions", any_sel, "captions.delete", json!({})),
        (Icon::Export, "text.captions.export", "Export captions…", !track.captions.is_empty(), "captions.export", json!({"track": track.id.0})),
    ];
    for (icon, id, label, enabled, cmd, params) in tools {
        let r = Rect::from_min_size(pos2(x, bar.min.y), vec2(24.0, 24.0));
        if r.max.x > rect.max.x {
            break;
        }
        if tool_button(app, ui, r, icon, id, label, enabled) {
            actions.push((cmd.into(), params));
        }
        x += 28.0;
    }

    // ---- style strip
    let style_h = 30.0;
    let style_rect = Rect::from_min_max(pos2(rect.min.x + 10.0, rect.max.y - style_h), pos2(rect.max.x - 10.0, rect.max.y - 2.0));
    style_strip(app, ui, style_rect, track_idx, &mut actions);

    // ---- segment list
    let list = Rect::from_min_max(pos2(rect.min.x + 6.0, bar.max.y + 8.0), pos2(rect.max.x - 6.0, style_rect.min.y - 6.0));
    ui.painter().rect_filled(list, 3.0, t.app_bg);
    let q = app.ui.caption_search.to_lowercase();
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(list.shrink(4.0)).id_salt("caption-list"));
    let current = track.caption_at(ph).map(|c| c.id);
    egui::ScrollArea::vertical().auto_shrink([false, false]).id_salt("caption-scroll").show(&mut child, |ui| {
        ui.set_width(list.width() - 12.0);
        let mut shown = 0;
        for (n, c) in track.captions.iter().enumerate() {
            if !q.is_empty() && !c.text.to_lowercase().contains(&q) && !c.speaker.as_deref().unwrap_or("").to_lowercase().contains(&q) {
                continue;
            }
            shown += 1;
            let selected = sel.contains(&c.id);
            let fill = if selected {
                t.row_selected
            } else if n % 2 == 1 {
                t.row_alt
            } else {
                Color32::TRANSPARENT
            };
            let frame = egui::Frame::NONE.fill(fill).inner_margin(egui::Margin::symmetric(6, 4)).corner_radius(3.0);
            let fr = frame.show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    ui.vertical(|ui| {
                        ui.set_width(92.0);
                        let num = ui.add(egui::Button::new(egui::RichText::new(format!("{}", n + 1)).size(11.0).color(t.text_dim)).frame(false));
                        app.auto.add(&format!("text.captions.{}.goto", c.id.0), num.rect, "Go to caption");
                        if num.clicked() {
                            if ui.input(|i| i.modifiers.command || i.modifiers.shift) {
                                actions.push(("captions.select".into(), json!({"captions": [c.id.0], "add": true})));
                            } else {
                                actions.push(("captions.goTo".into(), json!({"caption": c.id.0})));
                            }
                        }
                        for (edge, val) in [("in", c.start), ("out", c.end())] {
                            let key = egui::Id::new(("cap-tc", c.id.0, edge));
                            let mut buf = ui.data(|d| d.get_temp::<String>(key)).unwrap_or_else(|| tc(val));
                            let resp = ui.add(
                                egui::TextEdit::singleline(&mut buf)
                                    .id(key.with("te"))
                                    .desired_width(88.0)
                                    .font(Tokens::mono(11.0))
                                    .text_color(if edge == "in" { t.hot_text } else { t.text_dim }),
                            );
                            app.auto.add(&format!("text.captions.{}.{edge}", c.id.0), resp.rect, if edge == "in" { "Caption in" } else { "Caption out" });
                            if resp.has_focus() {
                                ui.data_mut(|d| d.insert_temp(key, buf.clone()));
                            } else {
                                ui.data_mut(|d| d.remove::<String>(key));
                            }
                            if resp.lost_focus() && buf.trim() != tc(val) {
                                let k = if edge == "in" { "startTimecode" } else { "endTimecode" };
                                actions.push(("captions.setTimes".into(), json!({"caption": c.id.0, k: buf.trim()})));
                            }
                        }
                    });
                    ui.vertical(|ui| {
                        if let Some(sp) = &c.speaker {
                            ui.label(egui::RichText::new(sp).size(11.0).color(t.text_dim).strong());
                        }
                        let key = egui::Id::new(("cap-text", c.id.0));
                        let mut buf = ui.data(|d| d.get_temp::<String>(key)).unwrap_or_else(|| c.text.clone());
                        let resp = ui.add(
                            egui::TextEdit::multiline(&mut buf)
                                .id(key.with("te"))
                                .desired_rows(1)
                                .desired_width(ui.available_width())
                                .font(Tokens::ui(12.5))
                                .frame(egui::Frame::NONE),
                        );
                        app.auto.add(&format!("text.captions.{}.text", c.id.0), resp.rect, "Caption text");
                        if resp.has_focus() {
                            ui.data_mut(|d| d.insert_temp(key, buf.clone()));
                            if !selected {
                                actions.push(("captions.select".into(), json!({"captions": [c.id.0]})));
                            }
                        } else {
                            ui.data_mut(|d| d.remove::<String>(key));
                        }
                        if resp.lost_focus() && buf != c.text {
                            actions.push(("captions.setText".into(), json!({"caption": c.id.0, "text": buf})));
                        }
                    });
                });
            });
            let row = fr.response.rect;
            if current == Some(c.id) {
                ui.painter().rect_stroke(row, 3.0, Stroke::new(1.0, t.accent), StrokeKind::Inside);
            }
            app.auto.add(&format!("text.captions.{}.row", c.id.0), row, &c.text);
            ui.add_space(2.0);
        }
        if shown == 0 {
            ui.label(
                egui::RichText::new(if q.is_empty() { "No captions on this track. Press + to add one at the playhead." } else { "No matching captions." })
                    .color(t.text_faint),
            );
        }
    });
    run(app, ui, actions);
}

/// Source clip currently targeted when viewing `Source` scope in the Transcript tab.
fn active_source_media_item(app: &FilmcraftApp) -> Option<ItemId> {
    let raw = app.session.state.source_item.or_else(|| app.session.state.project_selection.first().copied())?;
    filmcraft_engine::transcript::media_item(&app.session, raw)
}

fn transcript(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    let scope_id = egui::Id::new("text-transcript-scope"); // "sequence" | "source"
    let mode_id = egui::Id::new("text-transcript-mode"); // "edit" | "correct"
    let follow_id = egui::Id::new("text-transcript-follow");
    let show_replace_id = egui::Id::new("text-transcript-show-replace");
    let replace_buf_id = egui::Id::new("text-transcript-replace-buf");
    let word_edit_id = egui::Id::new("text-transcript-word-edit");
    let spk_edit_id = egui::Id::new("text-transcript-spk-edit");
    let drag_anchor_id = egui::Id::new("text-transcript-drag-anchor");

    let mut scope = ui.ctx().data(|d| d.get_temp::<String>(scope_id)).unwrap_or_else(|| "sequence".into());
    let mut edit_mode = ui.ctx().data(|d| d.get_temp::<String>(mode_id)).unwrap_or_else(|| "edit".into());
    let mut follow = ui.ctx().data(|d| d.get_temp::<bool>(follow_id)).unwrap_or(true);
    let mut show_replace = ui.ctx().data(|d| d.get_temp::<bool>(show_replace_id)).unwrap_or(false);
    let mut replace_buf = ui.ctx().data(|d| d.get_temp::<String>(replace_buf_id)).unwrap_or_default();

    let source_media = active_source_media_item(app);
    if scope == "source" && source_media.is_none() && app.session.active_sequence().is_some() {
        scope = "sequence".into();
    }

    if scope == "sequence" && app.session.active_sequence().is_none() {
        crate::dock::placeholder(ui, rect, &t, "Open a sequence to see its transcript");
        return;
    }

    let mut actions: Vec<(String, Value)> = Vec::new();
    let words: Vec<tx::SeqWord> = if scope == "source" {
        if let Some(item_id) = source_media
            && let Some(tr) = app.session.project.transcripts.get(&item_id)
        {
            tr.words
                .iter()
                .enumerate()
                .map(|(i, w)| tx::SeqWord {
                    text: w.text.clone(),
                    start: w.start,
                    end: w.end,
                    clip: ClipId(0),
                    item: item_id,
                    index: i,
                    track: 0,
                    speaker: tr.speaker_name(w),
                    confidence: w.confidence,
                })
                .collect()
        } else {
            Vec::new()
        }
    } else {
        filmcraft_engine::transcript::sequence_words(&app.session)
    };

    if words.is_empty() {
        let c = rect.center();
        icons::paint(ui.painter(), Rect::from_center_size(c - vec2(0.0, 76.0), vec2(40.0, 40.0)), Icon::Captions, t.text_dim);
        let title = if scope == "source" { "Transcribe source clip" } else { "Transcribe sequence" };
        ui.painter().text(c - vec2(0.0, 34.0), Align2::CENTER_CENTER, title, Tokens::semibold(16.0), t.text);
        let note = if filmcraft_speech_available(app) {
            "Descript-style text editing turns every spoken word and pause into an editable script."
        } else {
            "This build has no speech-to-text; import a transcript with transcript.set."
        };
        ui.painter().text(c - vec2(0.0, 12.0), Align2::CENTER_CENTER, note, Tokens::ui(12.0), t.text_dim);
        let r = Rect::from_center_size(c + vec2(0.0, 22.0), vec2(200.0, 28.0));
        let resp = ui.interact(r, egui::Id::new("text.transcript.generate"), Sense::click());
        ui.painter().rect_filled(r, 14.0, if resp.hovered() { t.accent_hover } else { t.accent });
        ui.painter().text(r.center(), Align2::CENTER_CENTER, "Transcribe", Tokens::semibold(12.0), Color32::WHITE);
        app.auto.add("text.transcript.generate", r, "Transcribe");
        if resp.clicked() {
            let params = match (scope.as_str(), source_media) {
                ("source", Some(id)) => json!({"items": [id.0]}),
                _ => json!({}),
            };
            actions.push(("transcript.generate".into(), params));
        }
        // Auto-transcribe toggle below button
        let auto_on = app.session.prefs.media_analysis.auto_transcribe;
        let ar = Rect::from_center_size(c + vec2(0.0, 58.0), vec2(220.0, 22.0));
        let a_label = if auto_on { "Auto-Transcribe Clips: ON" } else { "Enable Auto-Transcribe Clips" };
        if chip_button(app, ui, ar, "text.transcript.autoTranscribe", a_label, auto_on, "Automatically transcribe imported and sequence clips") {
            actions
                .push(("prefs.set".into(), json!({"values": {"mediaAnalysis.autoTranscribe": !auto_on, "mediaAnalysis.autoTranscribeScope": "allImported"}})));
            if !auto_on {
                actions.push(("transcript.generate".into(), json!({})));
            }
        }
        run(app, ui, actions);
        return;
    }

    let sel = app.ui.transcript_sel.filter(|(a, b)| *a < words.len() && *b < words.len());
    let (sa, sb) = sel.map(|(a, b)| (a.min(b), a.max(b))).unzip();
    let min_pause = Tick::from_seconds_f64(0.45);
    let stats = tx::script_stats(&words, min_pause);

    // ---- Row 1: Primary toolbar (Search, Extract, Lift, Split `/`, Fillers, Pauses, Captions, Retranscribe, Export)
    let bar = Rect::from_min_size(rect.min + vec2(10.0, 2.0), vec2(rect.width() - 20.0, 26.0));
    let sw = 165.0f32.min((bar.width() * 0.32).max(95.0));
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(Rect::from_min_size(bar.min, vec2(sw, 24.0))));
    let sresp = crate::widgets::search_field(&mut child, &mut app.ui.transcript_search, "Search script…", sw, &t);
    app.auto.add("text.transcript.search", sresp.rect, "Search transcript");
    let hits: Vec<std::ops::Range<usize>> = tx::search(&words, &app.ui.transcript_search);
    let mut x = bar.min.x + sw + 6.0;

    // Replace toggle button
    let rep_btn = Rect::from_min_size(pos2(x, bar.min.y + 1.0), vec2(24.0, 22.0));
    if chip_button(app, ui, rep_btn, "text.transcript.toggleReplace", "Aa", show_replace, "Find & Replace in Script") {
        show_replace = !show_replace;
        ui.ctx().data_mut(|d| d.insert_temp(show_replace_id, show_replace));
    }
    x += 28.0;

    let range = sel.map(|(a, b)| json!({"from": a.min(b), "to": a.max(b)}));
    let first_sel_word = sel.map(|(a, b)| a.min(b));
    let filler_tip = format!("Remove filler words ({} found)", stats.filler_count);
    let pause_tip = format!("Shorten pauses >= 0.45s ({} found)", stats.pause_count);
    let tools: [(Icon, &str, String, bool, &str, Value); 6] = [
        (
            Icon::Razor,
            "text.transcript.extract",
            "Extract selected words (Delete / Backspace)".into(),
            sel.is_some() && scope == "sequence",
            "transcript.extract",
            range.clone().unwrap_or_default(),
        ),
        (
            Icon::Trash,
            "text.transcript.lift",
            "Lift selected words (leave gap)".into(),
            sel.is_some() && scope == "sequence",
            "transcript.lift",
            range.unwrap_or_default(),
        ),
        (
            Icon::Marker,
            "text.transcript.splitAtWord",
            "Split clip / add Scene boundary at word (/)".into(),
            first_sel_word.is_some() && scope == "sequence",
            "transcript.splitAtWord",
            json!({"word": first_sel_word.unwrap_or(0)}),
        ),
        (Icon::Link, "text.transcript.removeFillers", filler_tip, scope == "sequence", "transcript.removeFillers", json!({})),
        (
            Icon::Pause,
            "text.transcript.removePauses",
            pause_tip,
            scope == "sequence",
            "transcript.removePauses",
            json!({"minSeconds": 0.45, "keepSeconds": 0.1}),
        ),
        (Icon::Captions, "text.transcript.createCaptions", "Create captions from script".into(), scope == "sequence", "transcript.createCaptions", json!({})),
    ];
    for (icon, id, label, enabled, cmd, params) in tools {
        let r = Rect::from_min_size(pos2(x, bar.min.y), vec2(24.0, 24.0));
        if r.max.x > rect.max.x - 56.0 {
            break;
        }
        if tool_button(app, ui, r, icon, id, &label, enabled) {
            if cmd.ends_with("extract") || cmd.ends_with("lift") {
                app.ui.transcript_sel = None;
            }
            actions.push((cmd.into(), params));
        }
        x += 26.0;
    }

    // Retranscribe (`text.transcript.generate`) and Export (`text.transcript.export`) icons on the right of the toolbar
    let gen_r = Rect::from_min_size(pos2(x, bar.min.y), vec2(24.0, 24.0));
    if gen_r.max.x <= rect.max.x && tool_button(app, ui, gen_r, Icon::Mic, "text.transcript.generate", "Transcribe / Retranscribe", true) {
        let params = match (scope.as_str(), source_media) {
            ("source", Some(id)) => json!({"items": [id.0]}),
            _ => json!({}),
        };
        actions.push(("transcript.generate".into(), params));
    }
    x += 26.0;
    let exp_r = Rect::from_min_size(pos2(x, bar.min.y), vec2(24.0, 24.0));
    if exp_r.max.x <= rect.max.x
        && tool_button(app, ui, exp_r, Icon::Export, "text.transcript.export", "Copy / Export Descript Markdown Script", true)
        && let Ok(v) = app.session.execute("transcript.export", json!({"format": "markdown"}))
        && let Some(txt) = v.get("text").and_then(Value::as_str)
    {
        ui.ctx().copy_text(txt.to_string());
        app.ui.status = format!("Copied script ({} words) to clipboard", words.len());
    }

    let mut top_y = bar.max.y + 4.0;

    // ---- Row 2: Descript Script Controls Bar (Scope, Cut Media vs Correct Text mode, Auto-Transcribe, Follow, Source Insert/Overwrite)
    let sub_bar = Rect::from_min_size(pos2(rect.min.x + 10.0, top_y), vec2(rect.width() - 20.0, 22.0));
    let mut sx = sub_bar.min.x;
    let r_seq = Rect::from_min_size(pos2(sx, sub_bar.min.y), vec2(64.0, 20.0));
    if chip_button(app, ui, r_seq, "text.transcript.scope.sequence", "Sequence", scope == "sequence", "View and edit active sequence script") {
        scope = "sequence".into();
        ui.ctx().data_mut(|d| d.insert_temp(scope_id, scope.clone()));
    }
    sx += 68.0;
    if source_media.is_some() {
        let r_src = Rect::from_min_size(pos2(sx, sub_bar.min.y), vec2(56.0, 20.0));
        if chip_button(
            app,
            ui,
            r_src,
            "text.transcript.scope.source",
            "Source",
            scope == "source",
            "View source clip transcript and insert phrases into timeline",
        ) {
            scope = "source".into();
            ui.ctx().data_mut(|d| d.insert_temp(scope_id, scope.clone()));
        }
        sx += 62.0;
    }

    // Mode toggle: Cut Media vs Correct Text
    let r_edit = Rect::from_min_size(pos2(sx, sub_bar.min.y), vec2(66.0, 20.0));
    if chip_button(
        app,
        ui,
        r_edit,
        "text.transcript.mode.edit",
        "Cut Media",
        edit_mode == "edit",
        "Edit mode: deleting words cuts the timeline (Descript style)",
    ) {
        edit_mode = "edit".into();
        ui.ctx().data_mut(|d| d.insert_temp(mode_id, edit_mode.clone()));
    }
    sx += 70.0;
    let r_corr = Rect::from_min_size(pos2(sx, sub_bar.min.y), vec2(78.0, 20.0));
    if chip_button(
        app,
        ui,
        r_corr,
        "text.transcript.mode.correct",
        "Correct Text",
        edit_mode == "correct",
        "Correct mode: clicking a word edits its spelling without cutting media",
    ) {
        edit_mode = "correct".into();
        ui.ctx().data_mut(|d| d.insert_temp(mode_id, edit_mode.clone()));
    }
    sx += 84.0;

    let auto_on = app.session.prefs.media_analysis.auto_transcribe;
    let r_auto = Rect::from_min_size(pos2(sx, sub_bar.min.y), vec2(96.0, 20.0));
    if r_auto.max.x <= sub_bar.max.x
        && chip_button(
            app,
            ui,
            r_auto,
            "text.transcript.autoTranscribe",
            if auto_on { "Auto-Transcribe ✓" } else { "Auto-Transcribe" },
            auto_on,
            "Automatically transcribe every imported or sequence video clip",
        )
    {
        actions.push(("prefs.set".into(), json!({"values": {"mediaAnalysis.autoTranscribe": !auto_on, "mediaAnalysis.autoTranscribeScope": "allImported"}})));
        if !auto_on {
            actions.push(("transcript.generate".into(), json!({})));
        }
    }
    sx += 100.0;

    let r_fol = Rect::from_min_size(pos2(sx, sub_bar.min.y), vec2(54.0, 20.0));
    if r_fol.max.x <= sub_bar.max.x && chip_button(app, ui, r_fol, "text.transcript.follow", "Follow", follow, "Follow active word during playback") {
        follow = !follow;
        ui.ctx().data_mut(|d| d.insert_temp(follow_id, follow));
    }
    sx += 58.0;

    if scope == "source"
        && let (Some(item_id), Some((a, b))) = (source_media, sel)
    {
        let (lo, hi) = (a.min(b), a.max(b));
        let r_ins = Rect::from_min_size(pos2(sx, sub_bar.min.y), vec2(52.0, 20.0));
        if r_ins.max.x <= sub_bar.max.x
            && chip_button(app, ui, r_ins, "text.transcript.insertSource", "Insert", true, "Insert selected source words into sequence at playhead")
        {
            actions.push(("transcript.insertFromSource".into(), json!({"item": item_id.0, "from": lo, "to": hi})));
        }
        sx += 56.0;
        let r_ovr = Rect::from_min_size(pos2(sx, sub_bar.min.y), vec2(66.0, 20.0));
        if r_ovr.max.x <= sub_bar.max.x
            && chip_button(app, ui, r_ovr, "text.transcript.overwriteSource", "Overwrite", false, "Overwrite selected source words into sequence at playhead")
        {
            actions.push(("transcript.overwriteFromSource".into(), json!({"item": item_id.0, "from": lo, "to": hi})));
        }
    }
    top_y = sub_bar.max.y + 4.0;

    // ---- Optional Row 3: Find & Replace Bar
    if show_replace {
        let rbar = Rect::from_min_size(pos2(rect.min.x + 10.0, top_y), vec2(rect.width() - 20.0, 24.0));
        let mut rchild = ui.new_child(egui::UiBuilder::new().max_rect(rbar).id_salt("transcript-replace-bar"));
        rchild.horizontal_centered(|ui| {
            ui.label(egui::RichText::new("Replace:").size(11.0).color(t.text_dim));
            let te = ui.add(egui::TextEdit::singleline(&mut replace_buf).hint_text("Replacement text…").desired_width(140.0).font(Tokens::ui(11.5)));
            app.auto.add("text.transcript.replaceInput", te.rect, "Replacement text");
            if te.changed() {
                ui.ctx().data_mut(|d| d.insert_temp(replace_buf_id, replace_buf.clone()));
            }
            let can_rep = !app.ui.transcript_search.trim().is_empty() && !hits.is_empty();
            let b1 = ui.add_enabled(can_rep, egui::Button::new("Replace"));
            app.auto.add("text.transcript.replaceOne", b1.rect, "Replace first match");
            if b1.clicked() {
                actions.push(("transcript.replace".into(), json!({"find": app.ui.transcript_search.trim(), "replace": replace_buf.trim(), "limit": 1})));
            }
            let b2 = ui.add_enabled(can_rep, egui::Button::new(format!("Replace All ({})", hits.len())));
            app.auto.add("text.transcript.replaceAll", b2.rect, "Replace all matches");
            if b2.clicked() {
                actions.push(("transcript.replace".into(), json!({"find": app.ui.transcript_search.trim(), "replace": replace_buf.trim()})));
            }
        });
        top_y = rbar.max.y + 4.0;
    }

    // ---- Keyboard shortcuts when no text field has focus
    let any_focused = ui.ctx().memory(|m| m.focused().is_some());
    if !any_focused
        && scope == "sequence"
        && let Some((a, b)) = sel
    {
        let (lo, hi) = (a.min(b), a.max(b));
        let shift = ui.input(|i| i.modifiers.shift);
        if ui.input(|i| i.key_pressed(egui::Key::Slash)) {
            actions.push(("transcript.splitAtWord".into(), json!({"word": lo})));
        } else if ui.input(|i| i.key_pressed(egui::Key::ArrowRight)) && hi + 1 < words.len() {
            let next = if shift { (a, (b + 1).min(words.len() - 1)) } else { (hi + 1, hi + 1) };
            app.ui.transcript_sel = Some(next);
            actions.push(("transcript.select".into(), json!({"from": next.0.min(next.1), "to": next.0.max(next.1)})));
        } else if ui.input(|i| i.key_pressed(egui::Key::ArrowLeft)) && lo > 0 {
            let prev = if shift { (a, b.saturating_sub(1)) } else { (lo - 1, lo - 1) };
            app.ui.transcript_sel = Some(prev);
            actions.push(("transcript.select".into(), json!({"from": prev.0.min(prev.1), "to": prev.0.max(prev.1)})));
        }
    }

    // ---- Descript-Style Script Document Area
    let list = Rect::from_min_max(pos2(rect.min.x + 6.0, top_y), pos2(rect.max.x - 6.0, rect.max.y - 4.0));
    ui.painter().rect_filled(list, 4.0, t.app_bg);
    ui.painter().rect_stroke(list, 4.0, Stroke::new(1.0, t.separator), StrokeKind::Inside);

    let ph = app.session.playhead();
    let current = tx::word_at(&words, ph);
    let paras = tx::paragraphs(&words, Tick::from_seconds_f64(1.5));
    let rate = app.session.sequence_rate();
    let df = app.session.active_sequence().is_some_and(|q| q.settings.drop_frame);

    let mut word_edit = ui.ctx().data(|d| d.get_temp::<Option<WordEditState>>(word_edit_id)).flatten();
    let mut spk_edit = ui.ctx().data(|d| d.get_temp::<Option<SpeakerEditState>>(spk_edit_id)).flatten();

    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(list.shrink(8.0)).id_salt("transcript-list"));
    egui::ScrollArea::vertical().auto_shrink([false, false]).id_salt("transcript-scroll").show(&mut child, |ui| {
        ui.set_width((list.width() - 20.0).max(80.0));

        // Script document metadata & stats strip (Descript header style)
        let doc_title = if scope == "source" {
            source_media.and_then(|id| app.session.project.item(id)).map(|i| i.name.clone()).unwrap_or_else(|| "Source Clip".into())
        } else {
            app.session.state.active_sequence.and_then(|id| app.session.project.item(id)).map(|i| i.name.clone()).unwrap_or_else(|| "Sequence".into())
        };
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new(doc_title).size(13.5).strong().color(t.text));
            ui.label(egui::RichText::new("·").color(t.text_faint));
            ui.label(egui::RichText::new(format!("{} words", stats.word_count)).size(11.0).color(t.text_dim));
            ui.label(egui::RichText::new(format!("{:.0} WPM", stats.wpm)).size(11.0).color(t.text_dim));
            if stats.filler_count > 0 {
                let lbl = egui::RichText::new(format!("{} filler{}", stats.filler_count, if stats.filler_count == 1 { "" } else { "s" }))
                    .size(11.0)
                    .color(Color32::from_rgb(235, 168, 52));
                if ui.add(egui::Button::new(lbl).frame(false)).on_hover_text("Click to remove all filler words").clicked() && scope == "sequence" {
                    actions.push(("transcript.removeFillers".into(), json!({})));
                }
            }
            if stats.pause_count > 0 {
                let lbl =
                    egui::RichText::new(format!("{} pause{}", stats.pause_count, if stats.pause_count == 1 { "" } else { "s" })).size(11.0).color(t.hot_text);
                if ui.add(egui::Button::new(lbl).frame(false)).on_hover_text("Click to shorten all pauses").clicked() && scope == "sequence" {
                    actions.push(("transcript.removePauses".into(), json!({"minSeconds": 0.45, "keepSeconds": 0.1})));
                }
            }
        });

        // Inline speaker rename bar when active
        if let Some(mut se) = spk_edit.clone() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("Rename {}:", se.old_name)).size(11.5).color(t.text_dim));
                let te = ui.add(egui::TextEdit::singleline(&mut se.buf).desired_width(130.0).font(Tokens::ui(12.0)));
                app.auto.add("text.transcript.renameSpeakerInput", te.rect, "Speaker name");
                let save = ui.button("Save").clicked() || (te.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));
                let cancel = ui.button("Cancel").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape));
                if save && !se.buf.trim().is_empty() {
                    actions.push(("transcript.renameSpeaker".into(), json!({"speaker": se.old_name, "name": se.buf.trim()})));
                    spk_edit = None;
                } else if cancel {
                    spk_edit = None;
                } else {
                    spk_edit = Some(se);
                }
            });
        }

        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);

        let mut scene_idx = 0usize;
        let mut prev_clip: Option<ClipId> = None;

        for (pi, pr) in paras.iter().enumerate() {
            if pr.is_empty() {
                continue;
            }
            let w0 = &words[pr.start];

            // Descript Scene `/` Boundary Divider whenever the underlying timeline clip changes
            if scope == "sequence" && prev_clip != Some(w0.clip) {
                scene_idx += 1;
                prev_clip = Some(w0.clip);
                let clip_name = app
                    .session
                    .active_sequence()
                    .and_then(|q| q.find_item(w0.clip))
                    .map(|(_, it)| it.name.clone())
                    .unwrap_or_else(|| format!("Clip {}", w0.clip.0));
                let tc_str = format_time(w0.start, rate, df, TimeDisplay::Timecode, 48_000);
                let scene_label = format!("/  Scene {scene_idx}  ·  {clip_name}  ·  {tc_str}");
                let s_resp =
                    ui.add(egui::Button::new(egui::RichText::new(&scene_label).size(11.0).color(t.text_dim).strong()).fill(t.panel_bg).corner_radius(3.0));
                app.auto.add(&format!("text.transcript.scene.{scene_idx}"), s_resp.rect, &scene_label);
                if s_resp.on_hover_text("Click to select scene clip on timeline").clicked() {
                    actions.push(("timeline.select".into(), json!({"clips": [w0.clip.0]})));
                    app.session.set_playhead(w0.start);
                }
                ui.add_space(4.0);
            }

            // Speaker badge + Timecode gutter (`text.transcript.paragraph.<pi>`)
            let sp_name = w0.speaker.as_deref().unwrap_or("Speaker");
            let sp_col = speaker_color(w0.speaker.as_deref());
            let tc_str = format_time(w0.start, rate, df, TimeDisplay::Timecode, 48_000);
            let head = format!("{sp_name}  {tc_str}");
            ui.horizontal(|ui| {
                let (dot_r, _) = ui.allocate_exact_size(vec2(8.0, 14.0), Sense::hover());
                ui.painter().circle_filled(dot_r.center(), 3.5, sp_col);
                let hr = ui.add(egui::Label::new(egui::RichText::new(&head).size(11.5).color(sp_col).strong()).sense(Sense::click()));
                app.auto.add(&format!("text.transcript.paragraph.{pi}"), hr.rect, "Paragraph");
                if hr.on_hover_text("Click to rename speaker or jump to paragraph").clicked() {
                    app.session.set_playhead(w0.start);
                    spk_edit = Some(SpeakerEditState { old_name: sp_name.to_string(), buf: sp_name.to_string() });
                }
            });

            // Paragraph word tokens & inline pause chips
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = vec2(4.5, 4.0);
                for i in pr.clone() {
                    let w = &words[i];

                    // Inline word text editor (double-click or Correct Text mode)
                    let is_editing_this = word_edit.as_ref().is_some_and(|we| we.word_idx == i);
                    if is_editing_this && let Some(mut we) = word_edit.clone() {
                        let desired_w = (we.buf.len() as f32 * 8.0 + 28.0).max(56.0);
                        let te = ui.add(
                            egui::TextEdit::singleline(&mut we.buf).id(egui::Id::new(("word-inline-te", i))).desired_width(desired_w).font(Tokens::ui(13.0)),
                        );
                        app.auto.add(&format!("text.transcript.word.{i}"), te.rect, &w.text);
                        app.auto.add("text.transcript.wordEdit", te.rect, "Edit word text");
                        let commit =
                            (te.lost_focus() && ui.input(|inp| !inp.key_pressed(egui::Key::Escape))) || ui.input(|inp| inp.key_pressed(egui::Key::Enter));
                        let cancel = ui.input(|inp| inp.key_pressed(egui::Key::Escape));
                        if commit {
                            let new_t = we.buf.trim().to_string();
                            if new_t != w.text {
                                if scope == "source"
                                    && let Some(item_id) = we.item
                                {
                                    actions.push(("transcript.editWord".into(), json!({"item": item_id, "itemWord": we.item_word, "text": new_t})));
                                } else {
                                    actions.push(("transcript.editWord".into(), json!({"word": i, "text": new_t})));
                                }
                            }
                            word_edit = None;
                        } else if cancel {
                            word_edit = None;
                        } else {
                            word_edit = Some(we);
                        }
                        continue;
                    }

                    let in_sel = sa.is_some_and(|a| i >= a) && sb.is_some_and(|b| i <= b);
                    let hit = hits.iter().any(|h| h.contains(&i));
                    let is_cur = Some(i) == current;
                    let is_filler = tx::is_filler_word(&w.text);

                    let fg = if is_cur {
                        t.hot_text
                    } else if is_filler {
                        Color32::from_rgb(235, 182, 84)
                    } else {
                        t.text
                    };
                    let mut text = egui::RichText::new(&w.text).size(13.5).color(fg);
                    if is_cur {
                        text = text.strong();
                    }
                    if in_sel {
                        text = text.background_color(t.row_selected);
                    } else if hit {
                        text = text.background_color(Color32::from_rgba_unmultiplied(220, 175, 45, 70));
                    }

                    let resp = ui.add(egui::Label::new(text).sense(Sense::click_and_drag()));
                    let wrect = resp.rect;

                    // Playhead karaoke cursor bar & underline decorations
                    if is_cur {
                        ui.painter()
                            .line_segment([pos2(wrect.min.x - 1.5, wrect.min.y + 1.0), pos2(wrect.min.x - 1.5, wrect.max.y - 1.0)], Stroke::new(2.0, t.accent));
                        ui.painter().line_segment([pos2(wrect.min.x, wrect.max.y), pos2(wrect.max.x, wrect.max.y)], Stroke::new(1.5, t.accent));
                        if follow && app.playback.playing {
                            resp.scroll_to_me(Some(egui::Align::Center));
                        }
                    } else if is_filler {
                        // Amber underline on filler words ("um", "uh"...)
                        ui.painter()
                            .line_segment([pos2(wrect.min.x, wrect.max.y), pos2(wrect.max.x, wrect.max.y)], Stroke::new(1.2, Color32::from_rgb(235, 168, 52)));
                    } else if w.confidence < 0.65 {
                        ui.painter().line_segment([pos2(wrect.min.x, wrect.max.y), pos2(wrect.max.x, wrect.max.y)], Stroke::new(1.0, t.text_faint));
                    }

                    app.auto.add(&format!("text.transcript.word.{i}"), wrect, &w.text);

                    // Double-click opens inline word correction
                    if resp.double_clicked() {
                        word_edit = Some(WordEditState { word_idx: i, item: Some(w.item.0), item_word: w.index, buf: w.text.clone() });
                    } else if resp.clicked() {
                        if edit_mode == "correct" {
                            word_edit = Some(WordEditState { word_idx: i, item: Some(w.item.0), item_word: w.index, buf: w.text.clone() });
                        } else {
                            let shift = ui.input(|inp| inp.modifiers.shift);
                            app.ui.transcript_sel = Some(match (shift, sel) {
                                (true, Some((a, _))) => (a, i),
                                _ => (i, i),
                            });
                            let (a, b) = app.ui.transcript_sel.unwrap_or((i, i));
                            if scope == "sequence" {
                                actions.push(("transcript.select".into(), json!({"from": a.min(b), "to": a.max(b)})));
                            }
                        }
                    }

                    // Drag-selection across words
                    if resp.drag_started() {
                        ui.ctx().data_mut(|d| d.insert_temp(drag_anchor_id, i));
                        app.ui.transcript_sel = Some((i, i));
                    }
                    if ui.input(|inp| inp.pointer.any_down())
                        && let Some(pos) = ui.input(|inp| inp.pointer.interact_pos())
                        && wrect.expand(2.0).contains(pos)
                        && let Some(anchor) = ui.ctx().data(|d| d.get_temp::<usize>(drag_anchor_id))
                        && app.ui.transcript_sel != Some((anchor, i))
                    {
                        app.ui.transcript_sel = Some((anchor, i));
                    }
                    if resp.drag_stopped()
                        && let Some((a, b)) = app.ui.transcript_sel
                    {
                        ui.ctx().data_mut(|d| d.remove::<usize>(drag_anchor_id));
                        if scope == "sequence" {
                            actions.push(("transcript.select".into(), json!({"from": a.min(b), "to": a.max(b)})));
                        }
                    }

                    // Right-click context menu on word
                    resp.context_menu(|ui| {
                        let (lo, hi) = app.ui.transcript_sel.map(|(a, b)| (a.min(b), a.max(b))).unwrap_or((i, i));
                        if ui.button("Correct Word Text…").clicked() {
                            word_edit = Some(WordEditState { word_idx: i, item: Some(w.item.0), item_word: w.index, buf: w.text.clone() });
                            ui.close();
                        }
                        if scope == "sequence" {
                            if ui.button("Extract Word(s) (Delete)").clicked() {
                                app.ui.transcript_sel = None;
                                actions.push(("transcript.extract".into(), json!({"from": lo, "to": hi})));
                                ui.close();
                            }
                            if ui.button("Lift Word(s) (Leave Gap)").clicked() {
                                app.ui.transcript_sel = None;
                                actions.push(("transcript.lift".into(), json!({"from": lo, "to": hi})));
                                ui.close();
                            }
                            ui.separator();
                            if ui.button("Split Scene Before Word (/)").clicked() {
                                actions.push(("transcript.splitAtWord".into(), json!({"word": i, "after": false})));
                                ui.close();
                            }
                            if ui.button("Split Scene After Word").clicked() {
                                actions.push(("transcript.splitAtWord".into(), json!({"word": i, "after": true})));
                                ui.close();
                            }
                            ui.separator();
                            if ui.button("Add Sequence Marker at Word").clicked() {
                                app.session.set_playhead(w.start);
                                actions.push(("markers.add".into(), json!({"name": w.text})));
                                ui.close();
                            }
                        } else if let Some(item_id) = source_media {
                            if ui.button("Insert into Sequence").clicked() {
                                actions.push(("transcript.insertFromSource".into(), json!({"item": item_id.0, "from": lo, "to": hi})));
                                ui.close();
                            }
                            if ui.button("Overwrite into Sequence").clicked() {
                                actions.push(("transcript.overwriteFromSource".into(), json!({"item": item_id.0, "from": lo, "to": hi})));
                                ui.close();
                            }
                        }
                    });

                    // Inline Descript Pause Pill `[0.8s]` when a pause >= min_pause follows this word
                    if let Some(next_w) = words.get(i + 1) {
                        let gap = next_w.start - w.end;
                        if gap >= min_pause {
                            let p_text = format!("⏸ {:.1}s", gap.seconds());
                            let p_resp =
                                ui.add(egui::Button::new(egui::RichText::new(&p_text).size(10.5).color(t.text_dim)).fill(t.field_bg).corner_radius(8.0));
                            app.auto.add(&format!("text.transcript.pause.{i}"), p_resp.rect, &p_text);
                            if p_resp.on_hover_text("Pause — click to ripple-delete pause").clicked() && scope == "sequence" {
                                actions.push(("transcript.deletePause".into(), json!({"afterWord": i, "keepSeconds": 0.05})));
                            }
                        }
                    }
                }
            });
            ui.add_space(10.0);
        }
    });

    ui.ctx().data_mut(|d| {
        d.insert_temp(word_edit_id, word_edit);
        d.insert_temp(spk_edit_id, spk_edit);
    });

    run(app, ui, actions);
}

/// The Graphics tab of the Text panel: lists all graphic text layers in the active sequence in
/// timeline order, with search, find-and-replace, inline editing (`graphics.setText`), and playhead
/// navigation.
fn graphics_tab(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    let Some(seq) = app.session.active_sequence().cloned() else {
        crate::dock::placeholder(ui, rect, &t, "Open a sequence to inspect graphic text layers");
        return;
    };

    let search_id = egui::Id::new("text-graphics-search");
    let replace_id = egui::Id::new("text-graphics-replace");
    let show_rep_id = egui::Id::new("text-graphics-show-rep");
    let mut query = ui.ctx().data(|d| d.get_temp::<String>(search_id)).unwrap_or_default();
    let mut replace_to = ui.ctx().data(|d| d.get_temp::<String>(replace_id)).unwrap_or_default();
    let mut show_replace = ui.ctx().data(|d| d.get_temp::<bool>(show_rep_id)).unwrap_or(false);
    let mut actions: Vec<(String, Value)> = Vec::new();

    // Collect all text layers across all video tracks of the active sequence
    struct GfxEntry {
        clip: ClipId,
        layer: usize,
        track_idx: usize,
        start: Tick,
        text: String,
        font: String,
        size: f64,
    }
    let mut entries: Vec<GfxEntry> = Vec::new();
    for (ti, tr) in seq.video_tracks.iter().enumerate() {
        for it in &tr.items {
            let Some(ItemKind::Graphic { width, height, .. }) = app.session.project.item(it.item).map(|p| &p.kind) else {
                continue;
            };
            let size = (*width, *height);
            let mt = it.source_time_at(it.start);
            for (li, &ei) in layer_indices(&it.effects).iter().enumerate() {
                if let Some(spec) = eval_layer(&it.effects[ei], mt, size)
                    && let LayerContent::Text(tp) = spec.content
                {
                    entries.push(GfxEntry { clip: it.id, layer: li, track_idx: ti, start: it.start, text: tp.text, font: tp.font, size: tp.size as f64 });
                }
            }
        }
    }
    entries.sort_by_key(|e| (e.start, e.track_idx, e.layer));

    // Top toolbar: Search, Find/Replace toggle, New Text
    let mut y_cursor = rect.min.y + 2.0;
    let bar = Rect::from_min_size(pos2(rect.min.x + 10.0, y_cursor), vec2(rect.width() - 20.0, 26.0));
    let mut bchild = ui.new_child(egui::UiBuilder::new().max_rect(bar).id_salt("text-gfx-bar"));
    bchild.horizontal_centered(|ui| {
        let sw = (bar.width() - 140.0).clamp(80.0, 180.0);
        let sresp = crate::widgets::search_field(ui, &mut query, "Search graphics…", sw, &t);
        app.auto.add("text.graphics.search", sresp.rect, "Search graphic text");
        if sresp.changed() {
            ui.ctx().data_mut(|d| d.insert_temp(search_id, query.clone()));
        }
        let rep_tog = ui.selectable_label(show_replace, "Aa").on_hover_text("Find & Replace in graphic text");
        app.auto.add("text.graphics.toggleReplace", rep_tog.rect, "Toggle Find & Replace");
        if rep_tog.clicked() {
            show_replace = !show_replace;
            ui.ctx().data_mut(|d| d.insert_temp(show_rep_id, show_replace));
        }
        let new_btn = ui.button("+ New Text");
        app.auto.add("text.graphics.newText", new_btn.rect, "New graphic text layer");
        if new_btn.clicked() {
            actions.push(("graphics.newText".into(), json!({"text": "New Title"})));
        }
    });
    y_cursor += 28.0;

    if show_replace {
        let rbar = Rect::from_min_size(pos2(rect.min.x + 10.0, y_cursor), vec2(rect.width() - 20.0, 26.0));
        let mut rchild = ui.new_child(egui::UiBuilder::new().max_rect(rbar).id_salt("text-gfx-repbar"));
        rchild.horizontal_centered(|ui| {
            ui.label(egui::RichText::new("Replace:").size(11.0).color(t.text_dim));
            let rresp = ui.add(egui::TextEdit::singleline(&mut replace_to).hint_text("Replacement…").desired_width(120.0).font(Tokens::ui(11.5)));
            app.auto.add("text.graphics.replaceInput", rresp.rect, "Replace graphic text");
            if rresp.changed() {
                ui.ctx().data_mut(|d| d.insert_temp(replace_id, replace_to.clone()));
            }
            let can_rep = !query.trim().is_empty() && entries.iter().any(|e| e.text.to_lowercase().contains(&query.trim().to_lowercase()));
            let rep_btn = ui.add_enabled(can_rep, egui::Button::new("Replace All"));
            app.auto.add("text.graphics.replaceAll", rep_btn.rect, "Replace all in graphics");
            if rep_btn.clicked() {
                let needle = query.trim();
                for e in &entries {
                    if e.text.to_lowercase().contains(&needle.to_lowercase()) {
                        let updated = replace_case_insensitive(&e.text, needle, replace_to.trim());
                        actions.push(("graphics.setText".into(), json!({"clip": e.clip.0, "layer": e.layer, "text": updated})));
                    }
                }
            }
        });
        y_cursor += 28.0;
    }

    if entries.is_empty() {
        let body = Rect::from_min_max(pos2(rect.min.x, y_cursor + 4.0), rect.max);
        let c = body.center();
        icons::paint(ui.painter(), Rect::from_center_size(c - vec2(0.0, 56.0), vec2(36.0, 36.0)), Icon::Type, t.text_dim);
        ui.painter().text(c - vec2(0.0, 20.0), Align2::CENTER_CENTER, "No graphic text in sequence", Tokens::semibold(15.0), t.text);
        ui.painter().text(c + vec2(0.0, 2.0), Align2::CENTER_CENTER, "Use the Type tool (T) or click + New Text to add titles.", Tokens::ui(12.0), t.text_dim);
        run(app, ui, actions);
        return;
    }

    let list = Rect::from_min_max(pos2(rect.min.x + 6.0, y_cursor + 4.0), pos2(rect.max.x - 6.0, rect.max.y - 4.0));
    ui.painter().rect_filled(list, 3.0, t.app_bg);
    let rate = seq.settings.frame_rate;
    let df = seq.settings.drop_frame;
    let q_low = query.trim().to_lowercase();

    let mut lchild = ui.new_child(egui::UiBuilder::new().max_rect(list.shrink(6.0)).id_salt("text-gfx-list"));
    egui::ScrollArea::vertical().auto_shrink([false, false]).id_salt("text-gfx-scroll").show(&mut lchild, |ui| {
        ui.set_width((list.width() - 16.0).max(80.0));
        for (idx, e) in entries.iter().enumerate() {
            if !q_low.is_empty() && !e.text.to_lowercase().contains(&q_low) && !e.font.to_lowercase().contains(&q_low) {
                continue;
            }
            let selected = app.session.state.selection.contains(&e.clip) && app.session.state.graphic_layers.contains(&e.layer);
            let fill = if selected {
                t.row_selected
            } else if idx % 2 == 1 {
                t.row_alt
            } else {
                Color32::TRANSPARENT
            };
            let frame = egui::Frame::NONE.fill(fill).inner_margin(egui::Margin::symmetric(8, 5)).corner_radius(3.0);
            let fr = frame.show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    let tc_in = format_time(e.start, rate, df, TimeDisplay::Timecode, 48_000);
                    let meta = format!("V{} · {tc_in}", e.track_idx + 1);
                    let go = ui.add(egui::Button::new(egui::RichText::new(meta).font(Tokens::mono(11.0)).color(t.hot_text)).frame(false));
                    if go.on_hover_text("Jump to graphic title").clicked() {
                        app.session.set_playhead(e.start);
                        actions.push(("graphics.selectLayer".into(), json!({"clip": e.clip.0, "layers": [e.layer]})));
                    }
                    ui.vertical(|ui| {
                        ui.label(egui::RichText::new(format!("{} · {:.0} px", e.font, e.size)).size(10.5).color(t.text_dim));
                        let key = egui::Id::new(("gfx-tab-te", e.clip.0, e.layer));
                        let mut buf = ui.data(|d| d.get_temp::<String>(key)).unwrap_or_else(|| e.text.clone());
                        let resp = ui.add(
                            egui::TextEdit::multiline(&mut buf)
                                .id(key.with("input"))
                                .desired_rows(1)
                                .desired_width(ui.available_width())
                                .font(Tokens::ui(12.5)),
                        );
                        app.auto.add(&format!("text.graphics.text.{}.{}", e.clip.0, e.layer), resp.rect, "Graphic text");
                        if resp.has_focus() {
                            ui.data_mut(|d| d.insert_temp(key, buf.clone()));
                        } else {
                            ui.data_mut(|d| d.remove::<String>(key));
                        }
                        if resp.lost_focus() && buf != e.text {
                            actions.push(("graphics.setText".into(), json!({"clip": e.clip.0, "layer": e.layer, "text": buf})));
                        }
                    });
                });
            });
            app.auto.add(&format!("text.graphics.row.{}.{}", e.clip.0, e.layer), fr.response.rect, &e.text);
            ui.add_space(2.0);
        }
    });

    run(app, ui, actions);
}

fn replace_case_insensitive(haystack: &str, needle: &str, replacement: &str) -> String {
    if needle.is_empty() {
        return haystack.to_string();
    }
    let h_low = haystack.to_lowercase();
    let n_low = needle.to_lowercase();
    if h_low.len() != haystack.len() || n_low.len() != needle.len() {
        return haystack.replace(needle, replacement);
    }
    let mut out = String::with_capacity(haystack.len());
    let mut last = 0;
    for (idx, _) in h_low.match_indices(&n_low) {
        out.push_str(&haystack[last..idx]);
        out.push_str(replacement);
        last = idx + needle.len();
    }
    out.push_str(&haystack[last..]);
    out
}

fn filmcraft_speech_available(app: &FilmcraftApp) -> bool {
    app.session.transcriber.is_some() || filmcraft_engine::transcript::speech_available()
}

fn style_strip(app: &mut FilmcraftApp, ui: &mut egui::Ui, r: Rect, track_idx: usize, actions: &mut Vec<(String, Value)>) {
    let Some(tr) = app.session.active_sequence().and_then(|q| q.caption_tracks.get(track_idx)).cloned() else { return };
    let st = tr.style.clone();
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(r).id_salt("caption-style"));
    child.horizontal_centered(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        let mut size = st.size;
        let resp = ui.add(egui::DragValue::new(&mut size).range(8.0..=200.0).speed(0.5).suffix(" px"));
        app.auto.add("text.captions.style.size", resp.rect, "Caption size");
        if resp.drag_stopped() || (resp.changed() && !resp.dragged()) {
            actions.push(("captions.setStyle".into(), json!({"track": tr.id.0, "size": size})));
        }
        let mut col = Color32::from_rgba_unmultiplied(st.color[0], st.color[1], st.color[2], st.color[3]);
        let resp = egui::color_picker::color_edit_button_srgba(ui, &mut col, egui::color_picker::Alpha::Opaque);
        app.auto.add("text.captions.style.color", resp.rect, "Text colour");
        if resp.changed() {
            let c = col.to_srgba_unmultiplied();
            actions.push(("captions.setStyle".into(), json!({"track": tr.id.0, "color": c})));
        }
        let mut bg = st.background;
        let resp = ui.checkbox(&mut bg, "Box");
        app.auto.add("text.captions.style.background", resp.rect, "Background box");
        if resp.changed() {
            actions.push(("captions.setStyle".into(), json!({"track": tr.id.0, "background": bg})));
        }
        ui.label(egui::RichText::new("Align").size(11.0).color(app.tokens.text_dim));
        for (a, icon_txt, name) in [(CaptionAlign::Left, "L", "left"), (CaptionAlign::Center, "C", "center"), (CaptionAlign::Right, "R", "right")] {
            let resp = ui.selectable_label(st.align == a, icon_txt).on_hover_text(format!("Align {name}"));
            app.auto.add(&format!("text.captions.style.align.{name}"), resp.rect, name);
            if resp.clicked() {
                actions.push(("captions.setStyle".into(), json!({"track": tr.id.0, "align": name})));
            }
        }
        ui.label(egui::RichText::new("Position").size(11.0).color(app.tokens.text_dim));
        for (a, name) in [(CaptionAnchor::Top, "top"), (CaptionAnchor::Middle, "middle"), (CaptionAnchor::Bottom, "bottom")] {
            let resp = ui.selectable_label(st.anchor == a, name[..1].to_uppercase()).on_hover_text(format!("Position: {name}"));
            app.auto.add(&format!("text.captions.style.anchor.{name}"), resp.rect, name);
            if resp.clicked() {
                actions.push(("captions.setStyle".into(), json!({"track": tr.id.0, "anchor": name})));
            }
        }
    });
}

fn run(app: &mut FilmcraftApp, ui: &egui::Ui, actions: Vec<(String, Value)>) {
    let ctx = ui.ctx().clone();
    for (cmd, p) in actions {
        if let Err(e) = crate::menus::invoke(app, &ctx, &cmd, p) {
            app.ui.status = e;
        }
    }
}
