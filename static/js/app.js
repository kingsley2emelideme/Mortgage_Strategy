/* =========================================================================
   Mortgage Rate Strategy — Frontend Application
   ========================================================================= */

'use strict';

// ---------------------------------------------------------------------------
// Chart colour palette (mirrors mortgage_analysis.py)
// ---------------------------------------------------------------------------
const C = {
  fixed:    '#1f4e79',
  variable: '#d4820a',
  hedged:   '#2e7d32',
  stress:   '#c62828',
  invest:   '#6a1b9a',
  fanFill:  'rgba(144,202,249,0.35)',
  fanMed:   '#0d47a1',
  grid:     '#e0e0e0',
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const charts = {};       // { chartId: Chart instance }
let lastResult = null;   // most recent API response

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const $ = id => document.getElementById(id);
const fmt = (n, dec = 2) => n == null ? '—' : n.toLocaleString('en-CA', { minimumFractionDigits: dec, maximumFractionDigits: dec });
const fmtPct = n => n == null ? '—' : n.toFixed(2) + '%';

function getInputs() {
  return {
    principal:           parseFloat($('principal').value),
    amortization_years:  parseInt($('amortization_years').value),
    payment_frequency:   $('payment_frequency').value,
    term_months:         parseInt($('term_months').value),
    fixed_rate:          parseFloat($('fixed_rate').value) / 100,
    var_rate_start:      parseFloat($('var_rate_start').value) / 100,
    lump_sum_amount:     parseFloat($('lump_sum_amount').value),
    lump_sum_month:      parseInt($('lump_sum_month').value),
    kappa:               parseFloat($('kappa').value),
    theta:               parseFloat($('theta').value) / 100,
    sigma:               parseFloat($('sigma').value) / 100,
    floor:               parseFloat($('floor').value) / 100,
    equity_cagr:         parseFloat($('equity_cagr').value) / 100,
    n_sims:              parseInt($('n_sims').value),
  };
}

// Destroy existing Chart.js instance before recreating
function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

// Build month labels: "M1" … "M60"
function monthLabels(months) {
  return months.map(m => `M${m}`);
}

// ---------------------------------------------------------------------------
// Export utilities
// ---------------------------------------------------------------------------
function exportCanvasPNG(canvasId, filename) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const link = document.createElement('a');
  link.download = filename + '.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
}

function exportCanvasJPEG(canvasId, filename) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const link = document.createElement('a');
  link.download = filename + '.jpeg';
  link.href = canvas.toDataURL('image/jpeg', 0.92);
  link.click();
}

async function exportCanvasPDF(canvasId, filename) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const { jsPDF } = window.jspdf;
  const imgData = canvas.toDataURL('image/png');
  const ratio = canvas.width / canvas.height;
  const pageW = 280, pageH = pageW / ratio;
  const pdf = new jsPDF({ orientation: ratio > 1 ? 'landscape' : 'portrait', unit: 'mm', format: 'a4' });
  pdf.addImage(imgData, 'PNG', 10, 10, pageW, pageH);
  pdf.save(filename + '.pdf');
}

function exportCanvasSVG(canvasId, filename) {
  // Fallback: export as high-res PNG with .svg extension note
  exportCanvasPNG(canvasId, filename + '_svg_equiv');
  alert('Note: Chart.js renders to Canvas. The exported PNG is equivalent to SVG quality at screen resolution.');
}

// Inject export button group into a .export-btns container
function injectExportButtons(container) {
  const chartId  = container.dataset.chart;
  const label    = container.dataset.label || chartId;
  container.innerHTML = `
    <div class="export-btn-group">
      <span class="text-muted small me-1 align-self-center">Export:</span>
      <button class="btn btn-outline-secondary btn-sm" onclick="exportCanvasPNG('${chartId}','${label}')">
        <i class="bi bi-filetype-png"></i> PNG
      </button>
      <button class="btn btn-outline-secondary btn-sm" onclick="exportCanvasJPEG('${chartId}','${label}')">
        <i class="bi bi-filetype-jpg"></i> JPEG
      </button>
      <button class="btn btn-outline-danger btn-sm" onclick="exportCanvasPDF('${chartId}','${label}')">
        <i class="bi bi-filetype-pdf"></i> PDF
      </button>
    </div>`;
}

// ---------------------------------------------------------------------------
// Chart builders
// ---------------------------------------------------------------------------

function buildBalanceChart(data) {
  destroyChart('chartBalance');
  const sc = data.schedules;
  const labels = monthLabels(sc.fixed.months);
  charts['chartBalance'] = new Chart($('chartBalance'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Fixed Baseline', data: sc.fixed.balance,    borderColor: C.fixed,    backgroundColor: 'transparent', tension: 0.3, borderWidth: 2, pointRadius: 0 },
        { label: 'Standard Variable', data: sc.variable.balance, borderColor: C.variable, backgroundColor: 'transparent', tension: 0.3, borderWidth: 2, pointRadius: 0 },
        { label: 'Hedged Var + LS', data: sc.hedged.balance,  borderColor: C.hedged,   backgroundColor: 'transparent', tension: 0.3, borderWidth: 2.5, pointRadius: 0 },
        { label: 'Stress (+2%)',   data: sc.stress.balance,   borderColor: C.stress,   backgroundColor: 'transparent', tension: 0.3, borderWidth: 1.5, borderDash: [5,3], pointRadius: 0 },
      ],
    },
    options: chartOptions('Outstanding Balance (CAD $)', true),
  });
}

function buildInterestChart(data) {
  destroyChart('chartInterest');
  const sc = data.schedules;
  const labels = monthLabels(sc.fixed.months);
  charts['chartInterest'] = new Chart($('chartInterest'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Fixed Baseline',  data: sc.fixed.cumulative_interest,    borderColor: C.fixed,    backgroundColor: 'transparent', tension: 0.3, borderWidth: 2, pointRadius: 0 },
        { label: 'Standard Variable',data: sc.variable.cumulative_interest, borderColor: C.variable, backgroundColor: 'transparent', tension: 0.3, borderWidth: 2, pointRadius: 0 },
        { label: 'Hedged Var + LS', data: sc.hedged.cumulative_interest,   borderColor: C.hedged,   backgroundColor: 'transparent', tension: 0.3, borderWidth: 2.5, pointRadius: 0 },
        { label: 'Stress (+2%)',    data: sc.stress.cumulative_interest,   borderColor: C.stress,   backgroundColor: 'transparent', tension: 0.3, borderWidth: 1.5, borderDash: [5,3], pointRadius: 0 },
      ],
    },
    options: chartOptions('Cumulative Interest Paid (CAD $)', true),
  });
}

function buildFanChart(data) {
  destroyChart('chartFan');
  const fc = data.fan_chart;
  const labels = monthLabels(fc.months);
  // Convert to percentage for display
  const toPercent = arr => arr.map(v => +(v * 100).toFixed(3));
  charts['chartFan'] = new Chart($('chartFan'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        // P10-P90 fill band
        { label: 'P90', data: toPercent(fc.p90), borderColor: 'transparent', backgroundColor: C.fanFill, fill: '+3', pointRadius: 0, tension: 0.3 },
        { label: 'P75', data: toPercent(fc.p75), borderColor: 'rgba(144,202,249,0.5)', backgroundColor: 'rgba(144,202,249,0.2)', fill: '+1', borderWidth: 1, pointRadius: 0, tension: 0.3 },
        { label: 'Median (P50)', data: toPercent(fc.p50), borderColor: C.fanMed, backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 0, tension: 0.3 },
        { label: 'P25', data: toPercent(fc.p25), borderColor: 'rgba(144,202,249,0.5)', backgroundColor: 'transparent', borderWidth: 1, pointRadius: 0, tension: 0.3 },
        { label: 'P10', data: toPercent(fc.p10), borderColor: 'transparent', backgroundColor: 'transparent', pointRadius: 0, tension: 0.3 },
      ],
    },
    options: chartOptions('Simulated Rate (%)', false, true),
  });
}

function buildMCChart(data) {
  destroyChart('chartMC');
  const mc = data.mc_histogram;
  const fixedLine = mc.centers.map(() => mc.fixed_total);
  const varLine   = mc.centers.map(() => mc.var_total);
  charts['chartMC'] = new Chart($('chartMC'), {
    data: {
      labels: mc.centers.map(v => '$' + fmt(v, 0)),
      datasets: [
        { type: 'bar', label: 'Scenario Count', data: mc.counts, backgroundColor: 'rgba(29,101,181,0.65)', borderColor: '#1d65b5', borderWidth: 1, yAxisID: 'y' },
        { type: 'line', label: 'Fixed Total Interest', data: fixedLine, borderColor: C.fixed, borderWidth: 2, borderDash: [6,3], pointRadius: 0, yAxisID: 'y2', tension: 0 },
        { type: 'line', label: 'Var Baseline', data: varLine, borderColor: C.variable, borderWidth: 2, borderDash: [3,3], pointRadius: 0, yAxisID: 'y2', tension: 0 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { boxWidth: 12, font: { size: 11 } } }, tooltip: { mode: 'index' } },
      scales: {
        x: { ticks: { maxRotation: 45, font: { size: 9 } }, grid: { color: C.grid } },
        y: { title: { display: true, text: 'Frequency', font: { size: 11 } }, grid: { color: C.grid } },
        y2: { position: 'right', title: { display: true, text: 'Total Interest (CAD $)', font: { size: 11 } }, grid: { drawOnChartArea: false }, ticks: { callback: v => '$' + fmt(v, 0) } },
      },
    },
  });
}

function buildOppChart(data) {
  destroyChart('chartOpp');
  const oc = data.opportunity_cost;
  const sc = data.schedules;
  const labels = monthLabels(oc.months);
  // Interest saving = fixed_cumulative - hedged_cumulative
  const interestSaving = sc.fixed.cumulative_interest.map((v, i) => +(v - sc.hedged.cumulative_interest[i]).toFixed(2));
  charts['chartOpp'] = new Chart($('chartOpp'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Investment Portfolio (Invest-the-Diff)', data: oc.portfolio, borderColor: C.invest, backgroundColor: 'rgba(106,27,154,0.08)', fill: true, tension: 0.3, borderWidth: 2.5, pointRadius: 0 },
        { label: 'Cumulative Interest Saving (Hedged vs Fixed)', data: interestSaving, borderColor: C.hedged, backgroundColor: 'transparent', tension: 0.3, borderWidth: 2, pointRadius: 0, borderDash: [4,3] },
      ],
    },
    options: chartOptions('Value (CAD $)', true),
  });
}

function buildConvexityChart(data) {
  destroyChart('chartConvexity');
  const cvx = data.convexity;
  charts['chartConvexity'] = new Chart($('chartConvexity'), {
    data: {
      labels: cvx.amounts.map(v => '$' + fmt(v, 0)),
      datasets: [
        { type: 'line', label: 'Total Interest Paid', data: cvx.total_interest, borderColor: C.fixed, backgroundColor: 'transparent', tension: 0.3, borderWidth: 2.5, pointRadius: 3, yAxisID: 'y' },
        { type: 'bar',  label: 'Marginal Saving per $5k', data: cvx.marginal_saving, backgroundColor: 'rgba(46,125,50,0.65)', borderColor: C.hedged, borderWidth: 1, yAxisID: 'y2' },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { boxWidth: 12, font: { size: 11 } } }, tooltip: { mode: 'index' } },
      scales: {
        x: { ticks: { maxRotation: 45, font: { size: 9 } }, grid: { color: C.grid } },
        y: { title: { display: true, text: 'Total Interest (CAD $)', font: { size: 11 } }, grid: { color: C.grid }, ticks: { callback: v => '$' + fmt(v, 0) } },
        y2: { position: 'right', title: { display: true, text: 'Marginal Saving ($)', font: { size: 11 } }, grid: { drawOnChartArea: false }, ticks: { callback: v => '$' + fmt(v, 0) } },
      },
    },
  });
}

// Shared chart options factory
function chartOptions(yLabel, dollarFormat = false, percentFormat = false) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { position: 'top', labels: { boxWidth: 12, font: { size: 11 } } },
      tooltip: {
        callbacks: {
          label: ctx => {
            const v = ctx.parsed.y;
            if (dollarFormat)  return ` ${ctx.dataset.label}: $${fmt(v)}`;
            if (percentFormat) return ` ${ctx.dataset.label}: ${fmtPct(v)}`;
            return ` ${ctx.dataset.label}: ${fmt(v)}`;
          },
        },
      },
    },
    scales: {
      x: { ticks: { maxTicksLimit: 12, font: { size: 10 } }, grid: { color: C.grid } },
      y: {
        title: { display: true, text: yLabel, font: { size: 11 } },
        grid: { color: C.grid },
        ticks: {
          callback: v => dollarFormat ? '$' + fmt(v, 0) : (percentFormat ? fmtPct(v) : fmt(v, 0)),
        },
      },
    },
  };
}

// ---------------------------------------------------------------------------
// KPI cards
// ---------------------------------------------------------------------------
function renderKPIs(summary, lumpSumMonth) {
  const row = $('kpiRow');
  const saving = summary.fixed_total_interest - summary.hedged_total_interest;
  row.innerHTML = `
    <div class="col-6 col-xl-3">
      <div class="kpi-card kpi-fixed">
        <div class="kpi-label"><i class="bi bi-lock-fill me-1"></i>Fixed Payment</div>
        <div class="kpi-value">$${fmt(summary.fixed_payment)}</div>
        <div class="kpi-sub">Total interest: $${fmt(summary.fixed_total_interest, 0)}</div>
      </div>
    </div>
    <div class="col-6 col-xl-3">
      <div class="kpi-card kpi-variable">
        <div class="kpi-label"><i class="bi bi-graph-up me-1"></i>Variable Payment</div>
        <div class="kpi-value">$${fmt(summary.var_payment)}</div>
        <div class="kpi-sub">Total interest: $${fmt(summary.var_total_interest, 0)}</div>
      </div>
    </div>
    <div class="col-6 col-xl-3">
      <div class="kpi-card kpi-hedged">
        <div class="kpi-label"><i class="bi bi-shield-check me-1"></i>Hedged Strategy</div>
        <div class="kpi-value">$${fmt(summary.hedged_total_interest, 0)}</div>
        <div class="kpi-sub">Interest saving vs fixed: $${fmt(saving, 0)}</div>
      </div>
    </div>
    <div class="col-6 col-xl-3">
      <div class="kpi-card kpi-savings">
        <div class="kpi-label"><i class="bi bi-graph-up-arrow me-1"></i>Portfolio (Invest-the-Diff)</div>
        <div class="kpi-value">$${fmt(summary.portfolio_terminal, 0)}</div>
        <div class="kpi-sub">Break-even inflation (fixed): ${fmtPct(summary.breakeven_inflation_fixed)}</div>
      </div>
    </div>`;
}

// ---------------------------------------------------------------------------
// Amortization table
// ---------------------------------------------------------------------------
function renderScheduleTable(rows, strategy, lumpSumMonth) {
  const tbody = $('scheduleBody');
  const strategyPmtMap = { fixed: rows[0] ? (rows[0].Interest + rows[0].Principal) : 0 };
  tbody.innerHTML = rows.map(row => {
    const pmt = (row.Interest + row.Principal).toFixed(2);
    const isLS = row.Month === lumpSumMonth && strategy === 'hedged';
    return `<tr class="${isLS ? 'lump-sum-row' : ''}">
      <td>${row.Month}${isLS ? ' <span class="badge bg-warning text-dark">LS</span>' : ''}</td>
      <td>${(row.Rate * 100).toFixed(3)}%</td>
      <td>$${fmt(parseFloat(pmt))}</td>
      <td>$${fmt(row.Interest)}</td>
      <td>$${fmt(row.Principal)}</td>
      <td>$${fmt(row.Balance)}</td>
      <td>$${fmt(row.CumulativeInterest)}</td>
      <td>$${fmt(row.CumulativePrincipal)}</td>
    </tr>`;
  }).join('');
}

// CSV export helper
function exportTableCSV() {
  const table = $('scheduleTable');
  if (!table) return;
  const rows = [];
  for (const row of table.querySelectorAll('tr')) {
    const cells = [...row.querySelectorAll('th,td')].map(c => `"${c.innerText.replace(/"/g, '""')}"`);
    rows.push(cells.join(','));
  }
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'amortization_schedule.csv';
  a.click();
}

async function exportTablePDF() {
  const { jsPDF } = window.jspdf;
  const strategy = $('scheduleStrategy').value;
  const canvas = await html2canvas($('scheduleTable'), { scale: 1.5 });
  const imgData = canvas.toDataURL('image/png');
  const pdf = new jsPDF({ orientation: 'landscape', unit: 'mm', format: 'a4' });
  const pageW = 277;
  const imgH = (canvas.height / canvas.width) * pageW;
  pdf.setFontSize(14);
  pdf.text(`Amortization Schedule — ${strategy.charAt(0).toUpperCase() + strategy.slice(1)} Strategy`, 10, 12);
  pdf.addImage(imgData, 'PNG', 10, 18, pageW, imgH);
  pdf.save(`amortization_${strategy}.pdf`);
}

// ---------------------------------------------------------------------------
// Main run function
// ---------------------------------------------------------------------------
async function runAnalysis() {
  const inputs = getInputs();

  // Show loading
  $('loadingOverlay').classList.remove('d-none');
  $('placeholderState').classList.add('d-none');

  try {
    const resp = await fetch('/api/calculate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputs),
    });

    if (!resp.ok) throw new Error(`Server error ${resp.status}`);
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    lastResult = data;

    // Show results panel
    $('resultsContainer').classList.remove('d-none');

    renderKPIs(data.summary, inputs.lump_sum_month);
    buildBalanceChart(data);
    buildInterestChart(data);
    buildFanChart(data);
    buildMCChart(data);
    buildOppChart(data);
    buildConvexityChart(data);

    // Inject export buttons into every .export-btns div
    document.querySelectorAll('.export-btns').forEach(injectExportButtons);

  } catch (err) {
    alert('Error running analysis:\n\n' + err.message);
  } finally {
    $('loadingOverlay').classList.add('d-none');
  }
}

// Load amortization table for selected strategy
async function loadSchedule() {
  if (!lastResult) { alert('Run the analysis first.'); return; }
  const inputs = getInputs();
  const strategy = $('scheduleStrategy').value;

  try {
    const resp = await fetch('/api/amortize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...inputs, strategy }),
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    renderScheduleTable(data.rows, strategy, inputs.lump_sum_month);
  } catch (err) {
    alert('Error loading schedule:\n\n' + err.message);
  }
}

// Reset to defaults
function resetInputs() {
  [['principal','750000'],['amortization_years','25'],['term_months','60'],
   ['fixed_rate','4.10'],['var_rate_start','3.35'],['lump_sum_amount','25000'],
   ['lump_sum_month','12'],['kappa','0.35'],['theta','3.5'],['sigma','1.2'],
   ['floor','2.25'],['equity_cagr','7.0']].forEach(([id, val]) => $(id).value = val);
  $('n_sims').value = '2000';
  $('payment_frequency').value = 'monthly';
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  $('btnRun').addEventListener('click', runAnalysis);
  $('btnRunBottom').addEventListener('click', runAnalysis);
  $('btnReset').addEventListener('click', () => { resetInputs(); });
  $('btnLoadSchedule').addEventListener('click', loadSchedule);
  $('btnExportCSV').addEventListener('click', exportTableCSV);
  $('btnExportTablePDF').addEventListener('click', exportTablePDF);

  // Auto-load schedule when strategy changes (if data is available)
  $('scheduleStrategy').addEventListener('change', () => {
    if (lastResult) loadSchedule();
  });
});
