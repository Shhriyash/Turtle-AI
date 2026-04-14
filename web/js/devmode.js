/**
 * devmode.js — Dev-mode sidebar panel logic
 *
 * Loads config + available models from the server, populates
 * dropdowns, sends updated config for hot-reload.
 */

import AppState from './state.js';
import { showToast } from './utils.js';

/** Toggle the dev panel open/close */
export function toggleDevPanel() {
    AppState.devPanelOpen = !AppState.devPanelOpen;
    AppState.dom.devSidebar.classList.toggle('open', AppState.devPanelOpen);
    AppState.dom.btnDevToggle.classList.toggle('active', AppState.devPanelOpen);

    if (AppState.devPanelOpen) {
        loadDevConfig();
    }
}

/** Load current config and model lists from the server */
export async function loadDevConfig() {
    try {
        const [modelsRes, configRes] = await Promise.all([
            fetch('/api/models'),
            fetch('/api/config'),
        ]);
        const models = await modelsRes.json();
        const cfg = await configRes.json();

        populateSelect('dev-openrouter-model', models.openrouter_models, cfg.OPEN_ROUTER_MODEL);
        populateSelect('dev-groq-model', models.groq_models, cfg.GROQ_PRIMARY_MODEL);
        populateSelect('dev-groq-fallback', models.groq_models, cfg.GROQ_FALLBACK_MODEL);
        populateSelect('dev-deepgram-model', models.deepgram_tts_models, cfg.DEEPGRAM_TTS_MODEL);
        populateSelect('dev-groq-voice', models.groq_tts_voices, cfg.GROQ_TTS_VOICE);

        document.getElementById('dev-temperature').value = cfg.temperature ?? 0.2;
        document.getElementById('dev-max-tokens').value = cfg.max_tokens ?? 1024;
        document.getElementById('dev-history-turns').value = cfg.TURTLE_HISTORY_MAX_TURNS ?? 12;
        document.getElementById('dev-max-messages').value = cfg.ACTIVE_HISTORY_MAX_MESSAGES ?? 40;
    } catch (e) {
        showToast('Failed to load config', true);
    }
}

/** Populate a <select> element with options */
function populateSelect(id, options, selected) {
    const select = document.getElementById(id);
    if (!select) return;
    select.innerHTML = '';
    for (const opt of options) {
        const el = document.createElement('option');
        el.value = opt;
        el.textContent = opt;
        if (opt === selected) el.selected = true;
        select.appendChild(el);
    }
}

/** Collect form values and POST to /api/config */
export async function applyDevConfig() {
    const cfg = {
        OPEN_ROUTER_MODEL: document.getElementById('dev-openrouter-model').value,
        GROQ_PRIMARY_MODEL: document.getElementById('dev-groq-model').value,
        GROQ_FALLBACK_MODEL: document.getElementById('dev-groq-fallback').value,
        DEEPGRAM_TTS_MODEL: document.getElementById('dev-deepgram-model').value,
        GROQ_TTS_VOICE: document.getElementById('dev-groq-voice').value,
        temperature: parseFloat(document.getElementById('dev-temperature').value),
        max_tokens: parseInt(document.getElementById('dev-max-tokens').value, 10),
        TURTLE_HISTORY_MAX_TURNS: parseInt(document.getElementById('dev-history-turns').value, 10),
        ACTIVE_HISTORY_MAX_MESSAGES: parseInt(document.getElementById('dev-max-messages').value, 10),
    };

    try {
        const res = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(cfg),
        });
        const result = await res.json();
        if (result.status === 'ok') {
            showToast('Config applied — agents reloaded');
        } else {
            showToast(result.error || 'Apply failed', true);
        }
    } catch (e) {
        showToast('Failed to apply config', true);
    }
}

/** Reset all fields to defaults and apply */
export async function resetDevDefaults() {
    const defaults = {
        OPEN_ROUTER_MODEL: 'nvidia/nemotron-3-nano-30b-a3b:free',
        GROQ_PRIMARY_MODEL: 'openai/gpt-oss-120b',
        GROQ_FALLBACK_MODEL: 'llama-3.1-8b-instant',
        DEEPGRAM_TTS_MODEL: 'aura-2-orion-en',
        GROQ_TTS_VOICE: 'orion',
        temperature: 0.2,
        max_tokens: 1024,
        TURTLE_HISTORY_MAX_TURNS: 12,
        ACTIVE_HISTORY_MAX_MESSAGES: 40,
    };

    try {
        const res = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(defaults),
        });
        const result = await res.json();
        if (result.status === 'ok') {
            showToast('Reset to defaults');
            loadDevConfig();
        }
    } catch (e) {
        showToast('Reset failed', true);
    }
}

/** Update the timing displays from a server timing message */
export function updateTimings(data) {
    // Show the panel timing bar
    const panelTiming = document.getElementById('panel-timing');
    if (panelTiming) panelTiming.classList.add('visible');

    if (data.stt_ms !== undefined) {
        const el = document.getElementById('timing-stt');
        if (el) el.textContent = data.stt_ms + 'ms';
    }
    if (data.llm_ms !== undefined) {
        const el = document.getElementById('timing-llm');
        if (el) el.textContent = data.llm_ms > 1000
            ? (data.llm_ms / 1000).toFixed(1) + 's'
            : data.llm_ms + 'ms';
    }
    if (data.tts_ms !== undefined) {
        const el = document.getElementById('timing-tts');
        if (el) el.textContent = data.tts_ms + 'ms';
    }
    if (data.total_ms !== undefined) {
        const el = document.getElementById('timing-total');
        if (el) el.textContent = data.total_ms > 1000
            ? (data.total_ms / 1000).toFixed(1) + 's'
            : data.total_ms + 'ms';
    }
}
