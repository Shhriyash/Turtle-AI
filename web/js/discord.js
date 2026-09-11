/**
 * discord.js — Discord Onboarding & Account Linking UI module
 */

import AppState from './state.js';
import { showToast } from './utils.js';

let modalBackdrop = null;
let modalCloseBtn = null;
let claimInput = null;
let submitBtn = null;
let statusMsg = null;
let headerBtn = null;

export function initDiscordUI() {
    modalBackdrop = document.getElementById('discord-modal');
    modalCloseBtn = document.getElementById('btn-discord-close');
    claimInput = document.getElementById('discord-claim-code');
    submitBtn = document.getElementById('btn-redeem-code');
    statusMsg = document.getElementById('discord-link-status');
    headerBtn = document.getElementById('btn-discord-toggle');

    if (headerBtn) {
        headerBtn.addEventListener('click', openDiscordModal);
    }

    const launchAppBtn = document.getElementById('btn-launch-discord-app');
    if (launchAppBtn) {
        launchAppBtn.addEventListener('click', () => {
            // Launch Discord desktop/mobile app directly via native URI scheme
            window.location.href = 'discord://users/1533903169866698983';
        });
    }

    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', closeDiscordModal);
    }

    if (modalBackdrop) {
        modalBackdrop.addEventListener('click', (e) => {
            if (e.target === modalBackdrop) {
                closeDiscordModal();
            }
        });
    }

    if (submitBtn) {
        submitBtn.addEventListener('click', handleRedeemCode);
    }

    if (claimInput) {
        claimInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                handleRedeemCode();
            }
        });
    }
}

export function openDiscordModal() {
    if (!modalBackdrop) return;
    modalBackdrop.classList.add('active');
    if (claimInput) {
        claimInput.value = '';
        claimInput.focus();
    }
    if (statusMsg) {
        statusMsg.className = 'discord-status-msg';
        statusMsg.style.display = 'none';
        statusMsg.textContent = '';
    }
}

export function closeDiscordModal() {
    if (!modalBackdrop) return;
    modalBackdrop.classList.remove('active');
}

async function handleRedeemCode() {
    if (!claimInput || !submitBtn) return;
    const rawCode = claimInput.value.trim();
    if (!rawCode) {
        showStatus('Please enter a valid claim code', 'error');
        return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = 'Linking...';
    showStatus('', 'none');

    try {
        const response = await fetch('/api/account/link', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ code: rawCode }),
        });

        const data = await response.json().catch(() => ({}));

        if (response.ok && data.status === 'ok') {
            if (data.already_linked) {
                showStatus('Account is already linked to your profile!', 'success');
            } else {
                showStatus('Successfully linked! Your Discord account and memories are now connected to Turtle Web.', 'success');
            }
            showToast('Discord account linked successfully!', false);
            claimInput.value = '';
        } else {
            const errorMsg = data.error || 'Failed to link account. Please check the code and try again.';
            showStatus(errorMsg, 'error');
        }
    } catch (err) {
        console.error('Account link error:', err);
        showStatus('Network error while linking account.', 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Link Account';
    }
}

function showStatus(msg, type) {
    if (!statusMsg) return;
    if (type === 'none' || !msg) {
        statusMsg.style.display = 'none';
        statusMsg.className = 'discord-status-msg';
        statusMsg.textContent = '';
        return;
    }
    statusMsg.textContent = msg;
    statusMsg.className = `discord-status-msg ${type}`;
    statusMsg.style.display = 'block';
}
