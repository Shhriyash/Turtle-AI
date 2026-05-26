/**
 * devmode.js — Dev-mode sidebar panel logic
 *
 * Loads config + available models from the server, populates
 * dropdowns, sends updated config for hot-reload.
 */

import AppState from './state.js';
import { showToast } from './utils.js';

function updateTtsSpeedLabel(value) {
    const num = Number(value);
    const el = document.getElementById('dev-tts-speed-value');
    if (el) el.textContent = Number.isFinite(num) ? num.toFixed(2) + 'x' : '1.20x';
}

function bindTtsSpeedPreview() {
    const slider = document.getElementById('dev-tts-speed');
    if (!slider || slider.dataset.bound === '1') return;
    slider.addEventListener('input', () => updateTtsSpeedLabel(slider.value));
    slider.dataset.bound = '1';
}

async function postConfigPatch(cfgPatch) {
    const res = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cfgPatch),
    });
    return await res.json();
}

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
        const [modelsRes, configRes, agentsRes] = await Promise.all([
            fetch('/api/models'),
            fetch('/api/config'),
            fetch('/api/agents'),
        ]);
        const models = await modelsRes.json();
        const cfg = await configRes.json();
        const agentsPayload = await agentsRes.json().catch(() => ({ agents: [] }));
        renderAgentsList(agentsPayload.agents || []);

        // Agent model overrides (prefixed: "groq:..." or "openrouter:...")
        populateSelect('dev-main-agent-model', models.all_models, cfg.MAIN_AGENT_MODEL);
        populateSelect('dev-email-agent-model', models.all_models, cfg.EMAIL_AGENT_MODEL);
        populateSelect('dev-dream-agent-model', models.all_models, cfg.DREAM_PASS_AGENT_MODEL);

        // Router uses AsyncGroq directly — Groq-only list, prefixed with "groq:" for consistency.
        const groqPrefixed = (models.groq_models || []).map(m => `groq:${m}`);
        populateSelect('dev-router-agent-model', groqPrefixed, cfg.ROUTER_AGENT_MODEL);

        // Dream Pass enabled toggle
        const dreamToggle = document.getElementById('dev-dream-pass-enabled');
        if (dreamToggle) dreamToggle.checked = !!cfg.PERSONAL_MEMORY_DREAM_PASS_ENABLED;

        // Fallback pools
        populateSelect('dev-openrouter-model', models.openrouter_models, cfg.OPEN_ROUTER_MODEL);
        populateSelect('dev-groq-model', models.groq_models, cfg.GROQ_PRIMARY_MODEL);
        populateSelect('dev-groq-fallback', models.groq_models, cfg.GROQ_FALLBACK_MODEL);

        // TTS
        populateSelect('dev-deepgram-model', models.deepgram_tts_models, cfg.DEEPGRAM_TTS_MODEL);
        populateSelect('dev-groq-voice', models.groq_tts_voices, cfg.GROQ_TTS_VOICE);

        // STT
        populateSelect('dev-stt-model', models.groq_stt_models, cfg.STT_MODEL);

        // Parameters
        document.getElementById('dev-temperature').value = cfg.temperature ?? 0.2;
        document.getElementById('dev-max-tokens').value = cfg.max_tokens ?? 1024;
        document.getElementById('dev-history-turns').value = cfg.TURTLE_HISTORY_MAX_TURNS ?? 12;
        document.getElementById('dev-max-messages').value = cfg.ACTIVE_HISTORY_MAX_MESSAGES ?? 40;
        document.getElementById('dev-tts-speed').value = cfg.TURTLE_TTS_SPEED ?? 1.2;
        updateTtsSpeedLabel(cfg.TURTLE_TTS_SPEED ?? 1.2);
        bindTtsSpeedPreview();
    } catch (e) {
        showToast('Failed to load config', true);
    }
}

/** Render the read-only runtime agents list */
function renderAgentsList(agents) {
    const container = document.getElementById('dev-agents-list');
    if (!container) return;
    container.innerHTML = '';
    for (const a of agents) {
        const row = document.createElement('div');
        row.className = 'dev-agent-row';
        const editableTag = a.editable ? '' : ' (read-only)';
        row.innerHTML = `
            <div class="dev-agent-label">${a.label}<span class="dev-agent-status" data-status="${a.status}">${a.status}${editableTag}</span></div>
            <div class="dev-agent-model">${a.model}</div>
        `;
        container.appendChild(row);
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
    // If the saved value isn't in the list, add it as the first option so it stays selected
    if (selected && !options.includes(selected)) {
        const el = document.createElement('option');
        el.value = selected;
        el.textContent = selected + ' (custom)';
        el.selected = true;
        select.insertBefore(el, select.firstChild);
    }
}

/** Collect form values and POST to /api/config */
export async function applyDevConfig() {
    const cfg = {
        MAIN_AGENT_MODEL: document.getElementById('dev-main-agent-model').value,
        EMAIL_AGENT_MODEL: document.getElementById('dev-email-agent-model').value,
        DREAM_PASS_AGENT_MODEL: document.getElementById('dev-dream-agent-model').value,
        ROUTER_AGENT_MODEL: document.getElementById('dev-router-agent-model').value,
        PERSONAL_MEMORY_DREAM_PASS_ENABLED: document.getElementById('dev-dream-pass-enabled').checked,
        OPEN_ROUTER_MODEL: document.getElementById('dev-openrouter-model').value,
        GROQ_PRIMARY_MODEL: document.getElementById('dev-groq-model').value,
        GROQ_FALLBACK_MODEL: document.getElementById('dev-groq-fallback').value,
        DEEPGRAM_TTS_MODEL: document.getElementById('dev-deepgram-model').value,
        TURTLE_TTS_SPEED: parseFloat(document.getElementById('dev-tts-speed').value),
        GROQ_TTS_VOICE: document.getElementById('dev-groq-voice').value,
        STT_MODEL: document.getElementById('dev-stt-model').value,
        temperature: parseFloat(document.getElementById('dev-temperature').value),
        max_tokens: parseInt(document.getElementById('dev-max-tokens').value, 10),
        TURTLE_HISTORY_MAX_TURNS: parseInt(document.getElementById('dev-history-turns').value, 10),
        ACTIVE_HISTORY_MAX_MESSAGES: parseInt(document.getElementById('dev-max-messages').value, 10),
    };

    try {
        const result = await postConfigPatch(cfg);
        if (result.status === 'ok') {
            showToast('Config applied — agents reloaded');
            loadDevConfig();
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
        MAIN_AGENT_MODEL: 'groq:openai/gpt-oss-120b',
        EMAIL_AGENT_MODEL: 'groq:llama-3.3-70b-versatile',
        DREAM_PASS_AGENT_MODEL: '',
        ROUTER_AGENT_MODEL: 'groq:llama-3.1-8b-instant',
        PERSONAL_MEMORY_DREAM_PASS_ENABLED: false,
        OPEN_ROUTER_MODEL: 'nvidia/llama-3.1-nemotron-70b-instruct:free',
        GROQ_PRIMARY_MODEL: 'llama-3.3-70b-versatile',
        GROQ_FALLBACK_MODEL: 'llama-3.1-8b-instant',
        DEEPGRAM_TTS_MODEL: 'aura-2-orion-en',
        TURTLE_TTS_SPEED: 1.2,
        GROQ_TTS_VOICE: 'orion',
        STT_MODEL: 'whisper-large-v3-turbo',
        temperature: 0.2,
        max_tokens: 1024,
        TURTLE_HISTORY_MAX_TURNS: 12,
        ACTIVE_HISTORY_MAX_MESSAGES: 40,
    };

    try {
        const result = await postConfigPatch(defaults);
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
