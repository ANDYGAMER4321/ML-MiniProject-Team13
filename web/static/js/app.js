"use strict";

/*
  Frontend for the Tennis Match Predictor.

  Talks to two Flask endpoints:
    GET  /api/players/search?q=...   -> list of players (array, or {players: [...]})
                                        each item is a name string or {name: "..."}
    POST /api/predict                -> { player1, player2, player1_probability,
                                          player2_probability, predicted_winner,
                                          threshold, historical_test_accuracy,
                                          notes? }
  On failure the API returns { error: "message" } with a non-2xx status.

  Also: GET /api/tournaments -> { tournaments: [...] }  (optional dropdown)
        GET /api/status      -> { data_through: "YYYY-MM-DD", ... }

  No probabilities or players are defined in this file: everything shown
  comes from the API response.
*/

const SEARCH_URL = "/api/players/search";
const PREDICT_URL = "/api/predict";
const MIN_QUERY_LENGTH = 2;
const DEBOUNCE_MS = 200;

const $ = (id) => document.getElementById(id);

const form = $("predict-form");
const surfaceEl = $("surface");
const dateEl = $("date");
const tournamentEl = $("tournament");
const errorEl = $("form-error");
const button = $("predict-btn");
const resultEl = $("result");

/* ---------------------------------------------------------------
   Player search boxes
   A player only counts as chosen when picked from the suggestions,
   so the backend always receives a name that exists in the data.
---------------------------------------------------------------- */
function normalisePlayers(payload) {
  const list = Array.isArray(payload) ? payload : (payload && payload.players) || [];
  return list
    .map((p) => (typeof p === "string" ? p : p && p.name))
    .filter((name) => typeof name === "string" && name.length > 0);
}

function createPlayerPicker(input, listEl) {
  const picker = { selected: null, input };
  let items = [];
  let active = -1;
  let timer = null;
  let controller = null;

  function open() { listEl.hidden = false; input.setAttribute("aria-expanded", "true"); }
  function close() { listEl.hidden = true; input.setAttribute("aria-expanded", "false"); active = -1; }

  function setActive(index) {
    active = index;
    [...listEl.children].forEach((li, i) => {
      li.setAttribute("aria-selected", i === active ? "true" : "false");
      if (i === active) li.scrollIntoView({ block: "nearest" });
    });
  }

  function render(names, message) {
    listEl.replaceChildren();
    items = names;
    if (names.length === 0) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = message;
      listEl.appendChild(li);
    } else {
      names.forEach((name, i) => {
        const li = document.createElement("li");
        li.setAttribute("role", "option");
        li.textContent = name;
        li.addEventListener("mousedown", (e) => { e.preventDefault(); choose(name); });
        listEl.appendChild(li);
      });
    }
    active = -1;
    open();
  }

  function choose(name) {
    picker.selected = name;
    input.value = name;
    input.classList.add("selected");
    close();
    clearError();
  }

  picker.set = (name) => {
    picker.selected = name || null;
    input.value = name || "";
    input.classList.toggle("selected", Boolean(name));
  };

  async function search(query) {
    if (controller) controller.abort();
    controller = new AbortController();
    try {
      const response = await fetch(`${SEARCH_URL}?q=${encodeURIComponent(query)}`, { signal: controller.signal });
      if (!response.ok) throw new Error("search failed");
      const names = normalisePlayers(await response.json());
      render(names, "No matching players found.");
    } catch (err) {
      if (err.name === "AbortError") return;
      render([], "Player search is unavailable right now.");
    }
  }

  input.addEventListener("input", () => {
    // Typing again invalidates a previous selection.
    picker.selected = null;
    input.classList.remove("selected");
    clearTimeout(timer);
    const query = input.value.trim();
    if (query.length < MIN_QUERY_LENGTH) { close(); return; }
    timer = setTimeout(() => search(query), DEBOUNCE_MS);
  });

  input.addEventListener("keydown", (e) => {
    if (listEl.hidden) return;
    if (e.key === "ArrowDown") { e.preventDefault(); setActive(Math.min(active + 1, items.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive(Math.max(active - 1, 0)); }
    else if (e.key === "Enter" && active >= 0) { e.preventDefault(); choose(items[active]); }
    else if (e.key === "Escape") { close(); }
  });

  input.addEventListener("blur", close);
  return picker;
}

const picker1 = createPlayerPicker($("player1"), $("list1"));
const picker2 = createPlayerPicker($("player2"), $("list2"));

$("swap").addEventListener("click", () => {
  const first = { text: picker1.input.value, chosen: picker1.selected };
  const second = { text: picker2.input.value, chosen: picker2.selected };
  picker1.set(second.chosen);
  picker2.set(first.chosen);
  // Keep half-typed (unchosen) text visible, but still unselected.
  if (!second.chosen) picker1.input.value = second.text;
  if (!first.chosen) picker2.input.value = first.text;
});

/* ---------------------------------------------------------------
   Date: default to today (local time)
---------------------------------------------------------------- */
function todayLocal() {
  const d = new Date();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${mm}-${dd}`;
}
dateEl.value = todayLocal();

/* ---------------------------------------------------------------
   Tournament list and data coverage (both optional: the page still
   works if these requests fail)
---------------------------------------------------------------- */
fetch("/api/tournaments")
  .then((r) => (r.ok ? r.json() : Promise.reject()))
  .then((data) => {
    (data.tournaments || []).forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      tournamentEl.appendChild(option);
    });
  })
  .catch(() => {});

fetch("/api/status")
  .then((r) => (r.ok ? r.json() : Promise.reject()))
  .then((s) => {
    if (s.data_through) {
      $("data-status").textContent = `Match data covers tournaments through ${s.data_through}. Predictions for later dates need newer data files.`;
    }
  })
  .catch(() => {});

/* ---------------------------------------------------------------
   Messages and loading state
---------------------------------------------------------------- */
function showError(message) { errorEl.textContent = message; errorEl.hidden = false; }
function clearError() { errorEl.hidden = true; errorEl.textContent = ""; }

function setLoading(isLoading) {
  button.disabled = isLoading;
  button.classList.toggle("loading", isLoading);
  button.querySelector(".btn-label").textContent = isLoading ? "Predicting…" : "Predict match";
}

/* ---------------------------------------------------------------
   Validation (the server validates again; this only saves a round trip)
---------------------------------------------------------------- */
function validate() {
  if (!picker1.selected) return "Choose Player 1 from the suggestions.";
  if (!picker2.selected) return "Choose Player 2 from the suggestions.";
  if (picker1.selected === picker2.selected) return "Choose two different players.";
  if (!surfaceEl.value) return "Choose a surface.";
  if (!dateEl.value || Number.isNaN(Date.parse(dateEl.value))) return "Enter a valid prediction date.";
  return null;
}

/* ---------------------------------------------------------------
   Result rendering. All text goes through textContent (no innerHTML).
---------------------------------------------------------------- */
function formatPercent(probability) { return `${(probability * 100).toFixed(2)}%`; }

function renderResult(data) {
  const p1 = Number(data.player1_probability);
  const p2 = Number(data.player2_probability);
  if (!Number.isFinite(p1) || !Number.isFinite(p2)) throw new Error("bad response");

  $("name1").textContent = data.player1;
  $("name2").textContent = data.player2;
  $("pct1").textContent = formatPercent(p1);
  $("pct2").textContent = formatPercent(p2);
  $("winner-name").textContent = data.predicted_winner;

  // Reset bars first so the width change animates on every prediction.
  $("bar1").style.width = "0%";
  $("bar2").style.width = "0%";
  resultEl.hidden = false;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $("bar1").style.width = `${p1 * 100}%`;
    $("bar2").style.width = `${p2 * 100}%`;
  }));

  const winnerProbability = data.predicted_winner === data.player1 ? p1 : p2;
  $("threshold-note").hidden = !(winnerProbability < 0.5);

  const notesEl = $("data-notes");
  notesEl.replaceChildren();
  const notes = Array.isArray(data.notes) ? data.notes : [];
  notes.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = String(text);
    notesEl.appendChild(li);
  });
  notesEl.hidden = notes.length === 0;

  if (data.threshold !== undefined) $("fact-threshold").textContent = String(data.threshold);
  if (data.historical_test_accuracy !== undefined) {
    $("fact-accuracy").textContent = `${Number(data.historical_test_accuracy).toFixed(2)}%`;
  }

  resultEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ---------------------------------------------------------------
   Submit
---------------------------------------------------------------- */
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();

  const problem = validate();
  if (problem) { showError(problem); return; }

  resultEl.hidden = true;
  setLoading(true);

  try {
    const response = await fetch(PREDICT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        player1: picker1.selected,
        player2: picker2.selected,
        surface: surfaceEl.value,
        date: dateEl.value,
        tournament: tournamentEl.value || null,
      }),
    });

    let data = null;
    try { data = await response.json(); } catch (_) { /* non-JSON error page */ }

    if (!response.ok) {
      showError((data && data.error) || "The server could not make a prediction. Please try again.");
      return;
    }
    renderResult(data);
  } catch (err) {
    showError(
      err instanceof TypeError
        ? "Can't reach the prediction server. Check your connection and try again."
        : "The server sent an unexpected response. Please try again."
    );
  } finally {
    setLoading(false);
  }
});
