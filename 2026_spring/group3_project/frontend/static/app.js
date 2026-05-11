const state = {
  session: null,
  markets: [],
  selectedMarketId: null,
  snapshot: null,
  portfolio: null,
  pollHandle: null,
  currentView: "market",
  lastRefreshAt: null,
};

const elements = {
  flash: document.getElementById("flash"),
  logoutButton: document.getElementById("logout-button"),
  sessionSummary: document.getElementById("session-summary"),
  signupForm: document.getElementById("signup-form"),
  loginForm: document.getElementById("login-form"),
  adminLoginForm: document.getElementById("admin-login-form"),
  tradeForm: document.getElementById("trade-form"),
  adminCreateMarketForm: document.getElementById("admin-create-market-form"),
  adminMarketStatusForm: document.getElementById("admin-market-status-form"),
  adminResolveMarketForm: document.getElementById("admin-resolve-market-form"),
  adminLoginShell: document.getElementById("admin-login-shell"),
  adminConsole: document.getElementById("admin-console"),
  accountChip: document.getElementById("account-chip"),
  marketList: document.getElementById("market-list"),
  marketSearch: document.getElementById("market-search"),
  metricLiveMarkets: document.getElementById("metric-live-markets"),
  metricSelectedMarket: document.getElementById("metric-selected-market"),
  metricLastRefresh: document.getElementById("metric-last-refresh"),
  marketTitle: document.getElementById("market-title"),
  marketDescription: document.getElementById("market-description"),
  marketStatusBadge: document.getElementById("market-status-badge"),
  marketPriceBadge: document.getElementById("market-price-badge"),
  yesBook: document.getElementById("yes-book"),
  noBook: document.getElementById("no-book"),
  balanceStrip: document.getElementById("balance-strip"),
  positionsList: document.getElementById("positions-list"),
  openOrdersList: document.getElementById("open-orders-list"),
  recentTradesList: document.getElementById("recent-trades-list"),
  walletEvents: document.getElementById("wallet-events"),
  marketChart: document.getElementById("history-chart"),
  walletChart: document.getElementById("wallet-chart"),
  adminMarketId: document.getElementById("admin-market-id"),
  adminResolveMarketId: document.getElementById("admin-resolve-market-id"),
  viewTabs: Array.from(document.querySelectorAll(".view-tab")),
  viewPanels: Array.from(document.querySelectorAll(".view-panel")),
};

function money(cents) {
  if (cents == null) return "-";
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(Number(cents) / 100);
}

function ts(value) {
  if (!value) return "-";
  return new Date(value).toLocaleString();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    credentials: "same-origin",
    ...options,
  });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function flash(message, type = "info") {
  elements.flash.textContent = message;
  elements.flash.classList.remove("hidden", "error");
  if (type === "error") {
    elements.flash.classList.add("error");
  } else {
    elements.flash.classList.remove("error");
  }
  window.clearTimeout(flash._timer);
  flash._timer = window.setTimeout(() => elements.flash.classList.add("hidden"), 3500);
}

function selectedMarket() {
  return state.markets.find((market) => market.market_id === state.selectedMarketId) || null;
}

function setView(view) {
  state.currentView = view;
  for (const tab of elements.viewTabs) {
    tab.classList.toggle("active", tab.dataset.view === view);
  }
  for (const panel of elements.viewPanels) {
    panel.classList.toggle("active", panel.id === `view-${view}`);
  }
  window.location.hash = view;
}

function syncViewFromHash() {
  const view = window.location.hash.replace(/^#/, "");
  if (["market", "portfolio", "wallet", "admin"].includes(view)) {
    setView(view);
  }
}

async function loadSession() {
  const payload = await api("/api/session");
  state.session = payload.session;
  renderSession();
}

async function loadMarkets() {
  const search = elements.marketSearch.value.trim();
  const suffix = search ? `?search=${encodeURIComponent(search)}` : "";
  const payload = await api(`/api/markets${suffix}`);
  state.markets = payload.markets || [];
  if (!state.selectedMarketId || !state.markets.some((market) => market.market_id === state.selectedMarketId)) {
    state.selectedMarketId = state.markets.length ? state.markets[0].market_id : null;
  }
  renderMarkets();
}

async function loadSnapshot() {
  if (!state.selectedMarketId) {
    state.snapshot = null;
    renderSnapshot();
    return;
  }
  const payload = await api(`/api/markets/${state.selectedMarketId}/snapshot`);
  state.snapshot = payload;
  renderSnapshot();
}

async function loadPortfolio() {
  if (!state.session) {
    state.portfolio = null;
    renderPortfolio();
    renderWallet();
    return;
  }
  const payload = await api("/api/portfolio");
  state.portfolio = payload;
  renderPortfolio();
  renderWallet();
}

async function refreshAll() {
  await loadSession();
  await loadMarkets();
  await loadSnapshot();
  await loadPortfolio();
  state.lastRefreshAt = new Date().toISOString();
  elements.metricLastRefresh.textContent = ts(state.lastRefreshAt);
}

function renderSession() {
  const session = state.session;
  elements.logoutButton.classList.toggle("hidden", !session);
  elements.adminLoginShell.classList.toggle("hidden", !!(session && session.is_admin));
  elements.adminConsole.classList.toggle("hidden", !(session && session.is_admin));

  if (!session) {
    elements.sessionSummary.textContent = "No active user session.";
    elements.accountChip.className = "account-chip empty";
    elements.accountChip.innerHTML = "Awaiting login";
    elements.balanceStrip.innerHTML = '<span class="data-pill">Authenticate to load balances</span>';
    return;
  }

  elements.sessionSummary.textContent = `${session.username} is active on account ${session.account_id}.`;
  elements.accountChip.className = `account-chip ${session.is_admin ? "admin" : "retail"}`;
  elements.accountChip.innerHTML = `
    <strong>${escapeHtml(session.username)}</strong>
    <span>${escapeHtml(session.account_id)}</span>
    <span>${session.is_admin ? "Admin" : "Retail"}</span>
  `;
}

function renderMarkets() {
  elements.metricLiveMarkets.textContent = String(state.markets.length);
  const currentMarket = selectedMarket();
  elements.metricSelectedMarket.textContent = currentMarket ? currentMarket.slug : "None";

  if (!state.markets.length) {
    elements.marketList.innerHTML = '<div class="empty-state">No markets matched your search.</div>';
    return;
  }

  elements.marketList.innerHTML = state.markets.map((market) => {
    const active = market.market_id === state.selectedMarketId ? "active" : "";
    return `
      <article class="market-card ${active}" data-market-id="${market.market_id}">
        <div class="market-card-top">
          <strong>${escapeHtml(market.title)}</strong>
          <span class="panel-tag compact">${escapeHtml(market.status)}</span>
        </div>
        <div class="market-card-prices">
          <span>YES ${market.live_yes_price == null ? "-" : `${market.live_yes_price}c`}</span>
          <span>NO ${market.live_no_price == null ? "-" : `${market.live_no_price}c`}</span>
        </div>
        <div class="market-card-meta">
          <span>OI ${market.open_interest_qty}</span>
          <span>${market.last_trade_at ? ts(market.last_trade_at) : "No prints"}</span>
        </div>
      </article>
    `;
  }).join("");

  for (const node of elements.marketList.querySelectorAll(".market-card")) {
    node.addEventListener("click", async () => {
      state.selectedMarketId = node.dataset.marketId;
      renderMarkets();
      await loadSnapshot();
    });
  }
}

function renderBook(container, rows, className) {
  if (!rows || !rows.length) {
    container.innerHTML = '<div class="empty-state">No resting liquidity.</div>';
    return;
  }
  container.innerHTML = rows.map((row) => `
    <div class="book-row ${className}">
      <strong>${row.price_cents}c</strong>
      <span>${row.qty} qty</span>
    </div>
  `).join("");
}

function drawSeries(canvas, points, { title, lineColor, fillColor, valueKey, valueFormatter }) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#081116";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(108, 133, 146, 0.22)";
  ctx.lineWidth = 1;
  for (let index = 0; index < 5; index += 1) {
    const y = 24 + ((height - 48) / 4) * index;
    ctx.beginPath();
    ctx.moveTo(24, y);
    ctx.lineTo(width - 24, y);
    ctx.stroke();
  }

  ctx.fillStyle = "#7e98a3";
  ctx.font = '16px "IBM Plex Mono", monospace';
  ctx.fillText(title, 28, 28);

  if (!points.length) {
    ctx.fillStyle = "#8ca0a9";
    ctx.fillText("No data yet", 28, height / 2);
    return;
  }

  const values = points.map((point) => Number(point[valueKey]));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1, max - min);
  const xScale = (index) => 28 + (index / Math.max(1, points.length - 1)) * (width - 56);
  const yScale = (value) => height - 28 - ((value - min) / span) * (height - 56);

  ctx.beginPath();
  points.forEach((point, index) => {
    const x = xScale(index);
    const y = yScale(Number(point[valueKey]));
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineWidth = 3;
  ctx.strokeStyle = lineColor;
  ctx.stroke();

  ctx.lineTo(xScale(points.length - 1), height - 28);
  ctx.lineTo(xScale(0), height - 28);
  ctx.closePath();
  ctx.fillStyle = fillColor;
  ctx.fill();

  const last = points[points.length - 1];
  ctx.fillStyle = "#d9e5ea";
  ctx.fillText(`${valueFormatter(Number(last[valueKey]))} | ${ts(last.created_at)}`, 28, 52);
}

function renderSnapshot() {
  const market = state.snapshot?.market || selectedMarket();
  if (!market) {
    elements.marketTitle.textContent = "Choose a market";
    elements.marketDescription.textContent = "Select a market from the tape to inspect price action and depth.";
    elements.marketStatusBadge.textContent = "No market";
    elements.marketPriceBadge.textContent = "-";
    elements.adminMarketId.value = "";
    elements.adminResolveMarketId.value = "";
    renderBook(elements.yesBook, [], "yes");
    renderBook(elements.noBook, [], "no");
    drawSeries(elements.marketChart, [], {
      title: "YES price history",
      lineColor: "#f3a52c",
      fillColor: "rgba(243, 165, 44, 0.18)",
      valueKey: "yes_price_cents",
      valueFormatter: (value) => `${value}c`,
    });
    return;
  }

  elements.marketTitle.textContent = market.title;
  elements.marketDescription.textContent = market.description || `Slug: ${market.slug}`;
  elements.marketStatusBadge.textContent = market.status;
  elements.marketPriceBadge.textContent = market.live_yes_price == null ? "No price" : `YES ${market.live_yes_price}c`;
  elements.adminMarketId.value = market.market_id;
  elements.adminResolveMarketId.value = market.market_id;
  renderBook(elements.yesBook, state.snapshot?.order_book?.yes || [], "yes");
  renderBook(elements.noBook, state.snapshot?.order_book?.no || [], "no");
  drawSeries(elements.marketChart, state.snapshot?.history?.points || [], {
    title: `${market.title} | YES price history`,
    lineColor: "#f3a52c",
    fillColor: "rgba(243, 165, 44, 0.18)",
    valueKey: "yes_price_cents",
    valueFormatter: (value) => `${value}c`,
  });
}

function renderPortfolio() {
  const portfolio = state.portfolio;
  if (!portfolio || !state.session) {
    elements.positionsList.innerHTML = '<div class="empty-state">No active session.</div>';
    elements.openOrdersList.innerHTML = '<div class="empty-state">No active session.</div>';
    elements.recentTradesList.innerHTML = '<div class="empty-state">No active session.</div>';
    return;
  }

  const balances = portfolio.balances;
  elements.balanceStrip.innerHTML = `
    <span class="data-pill">Available ${money(balances.available_cash_cents)}</span>
    <span class="data-pill">Locked ${money(balances.locked_cash_cents)}</span>
    <span class="data-pill">Deposits ${money(balances.total_deposits_cents)}</span>
    <span class="data-pill">Withdrawals ${money(balances.total_withdrawals_cents)}</span>
  `;

  elements.positionsList.innerHTML = portfolio.positions.length
    ? portfolio.positions.map((position) => `
        <div class="list-row">
          <div>
            <strong>${escapeHtml(position.market_title || position.market_id)}</strong>
            <div class="muted">YES ${position.yes_shares} | NO ${position.no_shares} | Mark ${position.live_yes_price == null ? "-" : `${position.live_yes_price}c`}</div>
          </div>
        </div>
      `).join("")
    : '<div class="empty-state">No positions yet.</div>';

  elements.openOrdersList.innerHTML = portfolio.open_orders.length
    ? portfolio.open_orders.map((order) => `
        <div class="list-row">
          <div>
            <strong>${escapeHtml(order.market_title || order.market_id)}</strong>
            <div class="muted">${order.side} ${order.remaining_qty}/${order.qty} @ ${order.price_cents}c</div>
          </div>
          <button class="ghost-button cancel-order" data-order-id="${order.request_id}" type="button">Cancel</button>
        </div>
      `).join("")
    : '<div class="empty-state">No open orders.</div>';

  elements.recentTradesList.innerHTML = portfolio.recent_trades.length
    ? portfolio.recent_trades.map((trade) => `
        <div class="list-row">
          <div>
            <strong>${escapeHtml(trade.market_title || trade.market_id)}</strong>
            <div class="muted">${trade.qty} contracts at ${trade.yes_price_cents}c | ${ts(trade.created_at)}</div>
          </div>
        </div>
      `).join("")
    : '<div class="empty-state">No trades yet.</div>';

  for (const button of elements.openOrdersList.querySelectorAll(".cancel-order")) {
    button.addEventListener("click", async () => {
      try {
        await api("/api/orders/cancel", {
          method: "POST",
          body: JSON.stringify({ order_id: button.dataset.orderId }),
        });
        flash("Order cancelled.");
        await loadPortfolio();
        await loadSnapshot();
      } catch (error) {
        flash(error.message, "error");
      }
    });
  }
}

function renderWallet() {
  const portfolio = state.portfolio;
  if (!portfolio || !state.session) {
    elements.walletEvents.innerHTML = '<div class="empty-state">No active session.</div>';
    drawSeries(elements.walletChart, [], {
      title: "Wallet cash over time",
      lineColor: "#4cd7c2",
      fillColor: "rgba(76, 215, 194, 0.18)",
      valueKey: "total_cash_cents",
      valueFormatter: (value) => money(value),
    });
    return;
  }

  const points = portfolio.wallet_history?.points || [];
  drawSeries(elements.walletChart, points, {
    title: `${portfolio.wallet_history?.username || state.session.username} | wallet cash`,
    lineColor: "#4cd7c2",
    fillColor: "rgba(76, 215, 194, 0.18)",
    valueKey: "total_cash_cents",
    valueFormatter: (value) => money(value),
  });

  elements.walletEvents.innerHTML = points.length
    ? [...points].reverse().slice(0, 20).map((point) => `
        <div class="list-row wallet-row">
          <div>
            <strong>${escapeHtml(point.reason)}</strong>
            <div class="muted">${point.notes ? escapeHtml(point.notes) : "No note"}</div>
          </div>
          <div class="wallet-meta">
            <span>${money(point.total_cash_cents)}</span>
            <span class="muted">${ts(point.created_at)}</span>
          </div>
        </div>
      `).join("")
    : '<div class="empty-state">No wallet activity yet.</div>';
}

async function handleAuthSubmit(event, path, successMessage) {
  event.preventDefault();
  const form = event.currentTarget;
  const formData = new FormData(form);
  try {
    await api(path, { method: "POST", body: JSON.stringify(Object.fromEntries(formData.entries())) });
    form.reset();
    flash(successMessage);
    await refreshAll();
  } catch (error) {
    flash(error.message, "error");
  }
}

async function handleTradeSubmit(event) {
  event.preventDefault();
  if (!state.selectedMarketId) {
    flash("Select a market first.", "error");
    return;
  }
  const payload = Object.fromEntries(new FormData(event.currentTarget).entries());
  payload.market_id = state.selectedMarketId;
  try {
    await api("/api/orders", { method: "POST", body: JSON.stringify(payload) });
    flash("Order submitted.");
    await loadSnapshot();
    await loadPortfolio();
    setView("portfolio");
  } catch (error) {
    flash(error.message, "error");
  }
}

async function handleLogout() {
  try {
    await api("/api/logout", { method: "POST", body: "{}" });
    flash("Logged out.");
    await refreshAll();
  } catch (error) {
    flash(error.message, "error");
  }
}

async function handleAdminCreateMarket(event) {
  event.preventDefault();
  try {
    await api("/api/admin/markets", { method: "POST", body: JSON.stringify(Object.fromEntries(new FormData(event.currentTarget).entries())) });
    flash("Market created.");
    event.currentTarget.reset();
    await loadMarkets();
    await loadSnapshot();
  } catch (error) {
    flash(error.message, "error");
  }
}

async function handleAdminStatus(event) {
  event.preventDefault();
  try {
    await api("/api/admin/markets/status", { method: "POST", body: JSON.stringify(Object.fromEntries(new FormData(event.currentTarget).entries())) });
    flash("Market status updated.");
    await loadMarkets();
    await loadSnapshot();
  } catch (error) {
    flash(error.message, "error");
  }
}

async function handleAdminResolve(event) {
  event.preventDefault();
  try {
    await api("/api/admin/markets/resolve", { method: "POST", body: JSON.stringify(Object.fromEntries(new FormData(event.currentTarget).entries())) });
    flash("Market resolved.");
    await loadMarkets();
    await loadSnapshot();
  } catch (error) {
    flash(error.message, "error");
  }
}

function startPolling() {
  window.clearInterval(state.pollHandle);
  state.pollHandle = window.setInterval(async () => {
    try {
      await refreshAll();
    } catch (error) {
      console.error(error);
    }
  }, 3000);
}

elements.marketSearch.addEventListener("input", async () => {
  try {
    await loadMarkets();
    await loadSnapshot();
  } catch (error) {
    flash(error.message, "error");
  }
});

elements.signupForm.addEventListener("submit", (event) => handleAuthSubmit(event, "/api/signup", "Account created."));
elements.loginForm.addEventListener("submit", (event) => handleAuthSubmit(event, "/api/login", "Trader session active."));
elements.adminLoginForm.addEventListener("submit", (event) => handleAuthSubmit(event, "/api/login", "Admin session active."));
elements.tradeForm.addEventListener("submit", handleTradeSubmit);
elements.logoutButton.addEventListener("click", handleLogout);
elements.adminCreateMarketForm.addEventListener("submit", handleAdminCreateMarket);
elements.adminMarketStatusForm.addEventListener("submit", handleAdminStatus);
elements.adminResolveMarketForm.addEventListener("submit", handleAdminResolve);

for (const tab of elements.viewTabs) {
  tab.addEventListener("click", () => setView(tab.dataset.view));
}

window.addEventListener("hashchange", syncViewFromHash);

(async function init() {
  syncViewFromHash();
  try {
    await refreshAll();
    startPolling();
  } catch (error) {
    flash(error.message, "error");
  }
})();
