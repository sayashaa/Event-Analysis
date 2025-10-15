let poller = null;
let refPoller = null;
let lastFrameSeen = 0;

// Reference frames
let referenceFrames = [];
let currentRefIndex = 0;

/* ----------------------------
   Real-time Table Polling
-----------------------------*/
function startPolling() {
    if (poller) return;
    const tbody = document.querySelector('#rtTable tbody');

    async function fetchAndAppend() {
        const url = lastFrameSeen === 0
            ? '/api/latest_rows'
            : `/api/latest_rows?after=${lastFrameSeen}`;

        try {
            const res = await fetch(url);
            const data = await res.json();
            if (!data.ok) {
                console.warn('Failed to fetch rows:', data.msg);
                return;
            }

            for (const r of data.rows) {
                const tr = document.createElement('tr');
                tr.innerHTML = `<td>${r.frame}</td><td>${r.id}</td><td>${r.x}</td><td>${r.y}</td>`;
                tbody.appendChild(tr);
                lastFrameSeen = Math.max(lastFrameSeen, Number(r.frame));
            }

            if (data.rows.length > 0) {
                const tableWrap = document.querySelector('.table-wrap');
                tableWrap.scrollTop = tableWrap.scrollHeight;
            }

        } catch (e) {
            console.error('Polling error:', e);
        }
    }

    poller = setInterval(fetchAndAppend, 800);
    fetchAndAppend();
}

/* ----------------------------
   Detection Controls
-----------------------------*/
async function startDetection() {
    const input = document.getElementById('videoPath');
    const path = input.value.trim();
    const msg = document.getElementById('msg');

    if (!path) {
        msg.textContent = 'Please enter a video path.';
        return;
    }

    try {
        const res = await fetch('/start_detection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_path: path })
        });
        const data = await res.json();

        if (!data.ok) {
            msg.textContent = data.msg || 'Failed to start.';
            return;
        }

        msg.textContent = 'Detection started.';

        document.querySelector('#rtTable tbody').innerHTML = '';
        lastFrameSeen = 0;

        const img = document.getElementById('preview');
        img.src = '/video_feed?' + Date.now();

        startPolling();
        startReferencePolling();
    } catch (err) {
        console.error('Error starting detection:', err);
        msg.textContent = 'Error starting detection.';
    }
}

async function pauseDetection() {
    try {
        const res = await fetch('/api/pause', { method: 'POST' });
        const data = await res.json();
        alert(data.msg || (data.ok ? 'Detection paused.' : 'Pause failed.'));
    } catch (err) {
        console.error('Pause error:', err);
    }
}

async function resumeDetection() {
    try {
        const res = await fetch('/api/resume', { method: 'POST' });
        const data = await res.json();
        alert(data.msg || (data.ok ? 'Detection resumed.' : 'Resume failed.'));
    } catch (err) {
        console.error('Resume error:', err);
    }
}

async function stopDetection() {
    try {
        const res = await fetch('/api/stop', { method: 'POST' });
        const data = await res.json();
        alert(data.msg || (data.ok ? 'Detection stopped.' : 'Stop failed.'));
    } catch (err) {
        console.error('Stop error:', err);
    }
    stopReferencePolling();
}

/* ----------------------------
   ID Reassign / Delete
-----------------------------*/
async function submitReassign() {
    const wrongId = document.getElementById('wrongId').value.trim();
    const correctId = document.getElementById('correctId').value.trim();
    const msg = document.getElementById('msg');

    if (!wrongId || !correctId) {
        msg.textContent = 'Please fill both Wrong ID and Correct ID.';
        return;
    }

    try {
        const res = await fetch('/api/reassign', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ wrong_id: wrongId, correct_id: correctId })
        });
        const data = await res.json();
        msg.textContent = data.msg || (data.ok ? 'Reassigned.' : 'Error.');
    } catch (err) {
        console.error('Reassign error:', err);
        msg.textContent = 'Error reassigning ID.';
    }
}

async function submitDelete() {
    const deleteId = document.getElementById('deleteId').value.trim();
    const deleteMsg = document.getElementById('deleteMsg');

    if (!deleteId) {
        deleteMsg.textContent = 'Please enter an ID to delete.';
        return;
    }

    try {
        const res = await fetch('/api/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: deleteId })
        });
        const data = await res.json();
        deleteMsg.textContent = data.msg || (data.ok ? 'Deleted.' : 'Error.');
    } catch (err) {
        console.error('Delete error:', err);
        deleteMsg.textContent = 'Error deleting ID.';
    }
}

/* ----------------------------
   Reference Frames Functions
-----------------------------*/
async function loadReferenceFrames() {
    try {
        const res = await fetch('/api/reference_frames');
        const data = await res.json();
        if (data.ok) {
            if (data.frames.length !== referenceFrames.length) {
                referenceFrames = data.frames;
                if (currentRefIndex >= referenceFrames.length) {
                    currentRefIndex = Math.max(0, referenceFrames.length - 1);
                }
                updateReferenceFrameDisplay();
            }
        }
    } catch (err) {
        console.error('Error loading reference frames:', err);
    }
}

function startReferencePolling() {
    if (refPoller) return;
    refPoller = setInterval(loadReferenceFrames, 2000);
    loadReferenceFrames();
}

function stopReferencePolling() {
    if (refPoller) {
        clearInterval(refPoller);
        refPoller = null;
    }
}

function updateReferenceFrameDisplay() {
    const imgEl = document.getElementById('refFrameImg');
    const label = document.getElementById('refFrameLabel');

    if (!referenceFrames || referenceFrames.length === 0) {
        imgEl.src = '';
        label.textContent = 'No reference frames yet';
        return;
    }

    imgEl.src = referenceFrames[currentRefIndex];
    label.textContent = `Reference ${currentRefIndex + 1} / ${referenceFrames.length}`;

    // Keep overlay synced if open
    const overlay = document.getElementById('refOverlay');
    const overlayImg = document.getElementById('refOverlayImg');
    if (overlay.classList.contains('shown')) {
        overlayImg.src = referenceFrames[currentRefIndex];
    }
}

function showNextReferenceFrame() {
    if (referenceFrames.length === 0) return;
    currentRefIndex = (currentRefIndex + 1) % referenceFrames.length;
    updateReferenceFrameDisplay();
}

function showPrevReferenceFrame() {
    if (referenceFrames.length === 0) return;
    currentRefIndex = (currentRefIndex - 1 + referenceFrames.length) % referenceFrames.length;
    updateReferenceFrameDisplay();
}

/* ----------------------------
   Reference Maximize / Minimize
-----------------------------*/
function openReferenceOverlay() {
    if (referenceFrames.length === 0) return;
    const overlay = document.getElementById('refOverlay');
    const overlayImg = document.getElementById('refOverlayImg');
    overlayImg.src = referenceFrames[currentRefIndex];
    overlay.classList.add('shown');
}

function closeReferenceOverlay() {
    const overlay = document.getElementById('refOverlay');
    overlay.classList.remove('shown');
}

/* ----------------------------
   Event Listeners
-----------------------------*/
window.addEventListener('DOMContentLoaded', () => {
    document.getElementById('startBtn').addEventListener('click', startDetection);
    document.getElementById('pauseBtn').addEventListener('click', pauseDetection);
    document.getElementById('resumeBtn').addEventListener('click', resumeDetection);
    document.getElementById('stopBtn').addEventListener('click', stopDetection);
    document.getElementById('submitBtn').addEventListener('click', submitReassign);
    document.getElementById('deleteBtn').addEventListener('click', submitDelete);

    // Reference navigation
    document.getElementById('refPrev').addEventListener('click', showPrevReferenceFrame);
    document.getElementById('refNext').addEventListener('click', showNextReferenceFrame);

    // Maximize / minimize overlay
    document.getElementById('refMaximize').addEventListener('click', openReferenceOverlay);
    document.getElementById('refOverlayClose').addEventListener('click', closeReferenceOverlay);
    document.getElementById('refOverlay').addEventListener('click', (e) => {
        if (e.target.id === 'refOverlay') closeReferenceOverlay();
    });
});
