const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const names = { CW: '未调制单频脉冲', LFM_UP: '线性上扫频', LFM_DOWN: '线性下扫频', NLFM: '非线性调频', BPSK: '二相编码', BFSK: '二频移键控', '4FSK': '四频移键控', SFM: '正弦调频' };
const state = { experiments: [], model: 'synthetic', view: 'recognize', input: 'sample', chart: 'spectrum', result: null, pulse: 0, file: null, busy: false, records: [] };
try { state.records = JSON.parse(localStorage.getItem('aimc-records') || '[]').filter(row => row && row.predictions).slice(0, 30); } catch { state.records = []; }
const icons = () => window.lucide?.createIcons();
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
const percent = (value, digits = 1) => (value * 100).toFixed(digits);
const experiment = () => state.experiments.find(row => row.id === state.model);

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '请求失败，请稍后重试。');
  return data;
}

function toast(message) {
  $('#toast').textContent = message;
  $('#toast').hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { $('#toast').hidden = true; }, 2800);
}

function download(data, filename, type = 'application/json') {
  const link = document.createElement('a');
  link.href = data.startsWith?.('data:') ? data : URL.createObjectURL(new Blob([data], { type }));
  link.download = filename;
  link.click();
  if (!link.href.startsWith('data:')) setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function saveRecords() {
  try { localStorage.setItem('aimc-records', JSON.stringify(state.records)); }
  catch { toast('浏览器存储已满，本次记录只保存在当前页面。'); }
  $('#history-count').textContent = state.records.length;
}

function recordResult(result) {
  // Keep only prediction metadata in browser storage, not uploaded IQ or images.
  const { spectrogram, waveform, ...record } = result;
  state.records.unshift(record);
  state.records = state.records.slice(0, 30);
  saveRecords();
}

function updateOverview() {
  const row = experiment();
  if (!row) return;
  $('#class-count').textContent = `${row.config.classes.length} CLASSES`;
  $('#metric-source').textContent = row.id === 'synthetic' ? '本地合成信号' : '官方数据 · 小规模演示';
  $('#accuracy-value').innerHTML = `${percent(row.metrics.accuracy, 2)}<span>%</span>`;
  $('#f1-value').textContent = row.metrics.macro_f1.toFixed(4);
  $('#test-count').textContent = `${row.metrics.samples.toLocaleString()} 条测试样本`;
  $('#sample-value').innerHTML = `${row.config.preprocess.n_samples.toLocaleString()} <span>点</span>`;
  $('#sample-rate-value').innerHTML = `<span>${row.config.preprocess.sample_rate_hz / 1e6} MHz 采样率</span><span>96 × 96 谱图</span>`;
  $('#sample-rate-input').value = row.config.preprocess.sample_rate_hz;
  $('#model-select').value = state.model;
  $('#model-note').innerHTML = `<span class="status-dot"></span>${row.checkpoint_available ? '本地权重可用' : '权重缺失'}<span>CPU</span>`;
  $('#experiment-notice').textContent = row.id === 'synthetic'
    ? '当前结果仅适用于本地合成信号分布。Softmax 分数未经置信度校准。'
    : '小规模演示，非官方基准协议。测试准确率仅 7.61%，当前权重不适合可靠识别。';
  $$('.experiment-select').forEach(select => { select.value = state.model; });
  const sampleTab = $('[data-input-tab="sample"]');
  sampleTab.disabled = state.model !== 'synthetic';
  sampleTab.title = state.model === 'synthetic' ? '' : '官方模型需上传 1000 点、50 MHz 的 IQ 信号';
  if (state.model === 'official') setInput('upload');
}

function switchModel(key) {
  if (state.busy || !state.experiments.some(row => row.id === key)) return;
  state.model = key;
  state.result = null;
  state.pulse = 0;
  updateOverview();
  $('#spectrum-image').removeAttribute('src');
  $('#waveform-canvas').getContext('2d').clearRect(0, 0, $('#waveform-canvas').width, $('#waveform-canvas').height);
  $('#chart-loading').innerHTML = '<i data-lucide="radio"></i><span>等待输入信号</span>';
  $('#chart-loading').hidden = false;
  $('#signal-filename').textContent = '等待信号';
  $('#prediction-result').innerHTML = '<div class="prediction-empty">等待识别结果</div>';
  $('#prediction-count').textContent = '0 条脉冲';
  $('#runtime').textContent = '—';
  $('#batch-results').innerHTML = '';
  $('#input-error').hidden = true;
  renderEvaluation();
  renderData();
  icons();
}

function setInput(input) {
  state.input = input;
  $$('.input-tab').forEach(button => {
    const active = button.dataset.inputTab === input;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', active);
  });
  $('#sample-input').hidden = input !== 'sample';
  $('#upload-input').hidden = input !== 'upload';
  $('#run-button span').textContent = input === 'sample' ? '生成并识别' : '开始识别';
  $('#input-error').hidden = true;
}

function setBusy(busy) {
  state.busy = busy;
  $$('#run-button, #model-select, .input-tab, #random-seed, #modulation-select, #seed-input, #snr-slider, .experiment-select').forEach(button => { button.disabled = busy; });
  if (!busy) $('[data-input-tab="sample"]').disabled = state.model === 'official';
  $('#run-button').innerHTML = busy ? '<span class="spinner"></span><span>正在识别…</span>' : `<i data-lucide="play"></i><span>${state.input === 'sample' ? '生成并识别' : '开始识别'}</span>`;
  if (busy) {
    $('#chart-loading').innerHTML = '<span class="spinner"></span><span>正在计算信号…</span>';
    $('#chart-loading').hidden = false;
  }
  icons();
}

async function run() {
  if (state.busy) return;
  $('#input-error').hidden = true;
  if (state.input === 'upload' && !state.file) {
    showError('请先选择一个 IQ 文件。');
    return;
  }
  const seed = Number($('#seed-input').value);
  if (state.input === 'sample' && (!Number.isInteger(seed) || seed < 0 || seed > 4294967295)) {
    showError('随机种子需为 0 到 4294967295 的整数。');
    return;
  }
  setBusy(true);
  try {
    let result;
    if (state.input === 'sample') {
      result = await api('/api/sample', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modulation: $('#modulation-select').value, snr: Number($('#snr-slider').value), seed }) });
    } else {
      const form = new FormData();
      form.append('model', state.model);
      form.append('file', state.file);
      form.append('sample_rate', $('#sample-rate-input').value);
      result = await api('/api/predict', { method: 'POST', body: form });
    }
    state.result = result;
    state.pulse = 0;
    recordResult(result);
    renderSignal();
    renderPrediction();
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
    $('#chart-loading').hidden = true;
  }
}

function showError(message) {
  $('#input-error').textContent = message;
  $('#input-error').hidden = false;
}

function renderSignal() {
  const result = state.result;
  if (!result) return;
  $('#spectrum-image').src = result.spectrogram;
  $('#signal-tag').textContent = result.ground_truth ? '样例信号' : '上传信号';
  $('#signal-filename').textContent = result.filename;
  $('#signal-details').textContent = `${result.n_samples} SAMPLES · ${(result.sample_rate_hz / 1e6).toFixed(0)} MHz`;
  $('#x-axis-ticks').innerHTML = [0, .25, .5, .75, 1].map(value => `<span>${Math.round(result.n_samples * value)}</span>`).join('');
  setChart(state.chart);
}

function setChart(chart) {
  state.chart = chart;
  $$('.signal-tab').forEach(button => button.classList.toggle('active', button.dataset.signalTab === chart));
  $('#spectrum-image').hidden = chart !== 'spectrum';
  $('#waveform-canvas').hidden = chart !== 'waveform';
  $('#color-scale').hidden = chart !== 'spectrum';
  if (chart === 'spectrum') {
    const fs = (state.result?.sample_rate_hz || experiment()?.config.preprocess.sample_rate_hz || 1e6) / 1e6;
    $('#y-axis-label').textContent = '频率 (MHz)';
    $('#y-axis-ticks').innerHTML = [.5, .25, 0, -.25, -.5].map(value => `<span>${(value * fs).toFixed(2)}</span>`).join('');
    $('#chart-format').textContent = 'FFT 64 / HOP 32';
  } else {
    $('#y-axis-label').textContent = '幅度';
    $('#chart-format').textContent = 'I / Q';
    if (state.result) drawWaveform();
  }
}

function canvasContext(canvas) {
  const rectangle = canvas.getBoundingClientRect();
  if (!rectangle.width || !rectangle.height) return null;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rectangle.width * ratio);
  canvas.height = Math.round(rectangle.height * ratio);
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  return { context, width: rectangle.width, height: rectangle.height };
}

function drawWaveform() {
  const chart = canvasContext($('#waveform-canvas'));
  if (!chart || !state.result) return;
  const { context: ctx, width, height } = chart;
  const values = state.result.waveform;
  const amplitude = Math.max(...values.i.map(Math.abs), ...values.q.map(Math.abs), 1e-6) * 1.1;
  ctx.fillStyle = '#12262c'; ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = '#2b3e42'; ctx.lineWidth = 1;
  for (let index = 0; index <= 4; index++) {
    ctx.beginPath(); ctx.moveTo(0, index * height / 4); ctx.lineTo(width, index * height / 4); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(index * width / 4, 0); ctx.lineTo(index * width / 4, height); ctx.stroke();
  }
  [values.i, values.q].forEach((series, index) => {
    ctx.strokeStyle = index === 0 ? '#60cba5' : '#dfb776'; ctx.lineWidth = 1.1; ctx.beginPath();
    series.forEach((value, point) => {
      const x = point / (series.length - 1) * width, y = height / 2 - value / amplitude * height / 2;
      if (point === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  ctx.font = '10px monospace'; ctx.fillStyle = '#60cba5'; ctx.fillText('I', 13, 18); ctx.fillStyle = '#dfb776'; ctx.fillText('Q', 40, 18);
  $('#y-axis-ticks').innerHTML = [1, .5, 0, -.5, -1].map(value => `<span>${(value * amplitude).toFixed(1)}</span>`).join('');
}

function renderPrediction() {
  const result = state.result;
  if (!result) return;
  const prediction = result.predictions[state.pulse];
  const uncertain = prediction.confidence < Number($('#confidence-slider').value) / 100;
  const entries = Object.entries(prediction.scores).sort((a, b) => b[1] - a[1]);
  $('#prediction-count').textContent = `${result.predictions.length} 条脉冲`;
  $('#runtime').textContent = `${result.duration_ms} ms · CPU`;
  const truth = result.ground_truth;
  const detail = uncertain ? '低于置信度阈值 · 待确认' : (names[prediction.best_class] || '官方调制类别');
  $('#prediction-result').innerHTML = `<div class="prediction-top ${uncertain ? 'uncertain' : ''}"><div class="prediction-identity"><span class="prediction-symbol"><i data-lucide="${uncertain ? 'circle-help' : 'scan-line'}"></i></span><div><strong>${escapeHtml(uncertain ? '待确认' : prediction.best_class)}</strong><p>${escapeHtml(detail)}${truth ? ` · 真实标签 ${escapeHtml(truth)}` : ''}</p></div></div><div class="prediction-confidence"><strong>${percent(prediction.confidence)}<span>%</span></strong><p>模型分数</p></div></div><div class="probabilities">${entries.slice(0, 5).map(([name, score]) => `<div class="prob-row"><span>${escapeHtml(name)}</span><div class="prob-track"><div class="prob-fill" style="width:${percent(score, 4)}%"></div></div><span>${percent(score)}%</span></div>`).join('')}</div>${entries.length > 5 ? `<p class="prob-more">其余 ${entries.length - 5} 类合计 ${percent(entries.slice(5).reduce((sum, row) => sum + row[1], 0))}%</p>` : ''}`;
  if (result.predictions.length > 1) {
    $('#batch-results').innerHTML = `<div class="batch-heading"><span>批量预测 · 当前结果 #${state.pulse + 1}（图表显示首条脉冲）</span><select class="text-input" id="pulse-select" aria-label="选择脉冲">${result.predictions.map((row, index) => `<option value="${index}" ${index === state.pulse ? 'selected' : ''}>脉冲 #${index + 1}</option>`).join('')}</select></div>`;
    $('#pulse-select').addEventListener('change', event => { state.pulse = Number(event.target.value); renderPrediction(); });
  }
  icons();
}

function changeView(view) {
  state.view = view;
  const content = { recognize: ['信号识别', '雷达信号识别', '单脉冲分析 / STFT 功率谱 / 调制样式识别'], evaluation: ['实验评估', '实验评估', '独立测试指标 / 训练过程 / 分类别表现'], models: ['模型与数据', '模型与数据', '本地权重 / 数据划分 / 预处理参数'], history: ['识别记录', '识别记录', '当前浏览器的最近预测结果'] }[view];
  $('#breadcrumb').textContent = content[0]; $('#page-title').textContent = content[1]; $('#page-subtitle').textContent = content[2];
  $$('.nav-item').forEach(button => button.classList.toggle('active', button.dataset.view === view));
  $$('.view').forEach(section => section.classList.toggle('active', section.id === `view-${view}`));
  $('#export-button').hidden = view === 'models';
  if (view === 'evaluation') renderEvaluation();
  if (view === 'history') renderHistory();
  if (view === 'models') renderData();
  if (view === 'recognize' && state.chart === 'waveform') drawWaveform();
  window.scrollTo({ top: 0, behavior: 'instant' });
}

function renderEvaluation() {
  const row = experiment();
  if (!row) return;
  $('#snr-bars').innerHTML = Object.entries(row.metrics.accuracy_by_snr_db).sort((a, b) => Number(a[0]) - Number(b[0])).map(([snr, score]) => `<div class="snr-row"><span>${Number(snr) > 0 ? '+' : ''}${Number(snr)} dB</span><div class="prob-track"><div class="prob-fill" style="width:${percent(score)}%;background:#499b7b"></div></div><span>${percent(score)}%</span></div>`).join('');
  $('#class-metrics').innerHTML = Object.entries(row.metrics.per_class).map(([name, metric]) => `<tr><td>${escapeHtml(name)}</td><td>${metric.precision.toFixed(4)}</td><td>${metric.recall.toFixed(4)}</td><td>${metric.f1.toFixed(4)}</td><td>${metric.support}</td></tr>`).join('');
  $('#protocol-note').textContent = `最佳权重：第 ${row.metrics.best_epoch} 轮，按验证损失选取。${row.id === 'official' ? '官方测试文件按原始脉冲重新划分，所有 SNR 版本留在同一集合；本实验非官方固定测试协议。' : '训练、验证与测试信号独立生成，测试集不用于选择权重。'}`;
  if (state.view === 'evaluation') drawHistory();
}

function drawHistory() {
  const chart = canvasContext($('#history-canvas'));
  if (!chart) return;
  const { context: ctx, width, height } = chart;
  const history = experiment().history;
  const margin = { left: 31, right: 10, top: 10, bottom: 24 };
  const w = width - margin.left - margin.right, h = height - margin.top - margin.bottom;
  ctx.font = '9px monospace';
  for (let index = 0; index <= 4; index++) {
    const y = margin.top + index * h / 4;
    ctx.strokeStyle = '#e8eee6'; ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(width - margin.right, y); ctx.stroke();
    ctx.fillStyle = '#9ba994'; ctx.fillText(String(100 - index * 25), 0, y + 3);
  }
  [1, Math.ceil(history.length / 2), history.length].forEach(epoch => {
    ctx.fillText(String(epoch), margin.left + (epoch - 1) / (history.length - 1) * w - 3, height - 5);
  });
  ['train_accuracy', 'val_accuracy'].forEach((field, index) => {
    ctx.strokeStyle = index === 0 ? '#138579' : '#d5a35a'; ctx.lineWidth = 2; ctx.beginPath();
    history.forEach((row, point) => {
      const x = margin.left + point / (history.length - 1) * w, y = margin.top + (1 - row[field]) * h;
      if (point === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }); ctx.stroke();
  });
}

function renderModels() {
  $('#model-list').innerHTML = state.experiments.map(row => `<div class="model-row"><span class="model-icon"><i data-lucide="box"></i></span><div><h3>STFT-CNN · ${escapeHtml(row.name)}</h3><p>${escapeHtml(row.window)} · 最佳权重第 ${row.metrics.best_epoch} 轮 · ${row.id === 'synthetic' ? '合成信号实验' : '小规模流程演示'}</p></div><button class="button secondary" data-use-model="${row.id}"><i data-lucide="arrow-up-right"></i>使用模型</button></div>`).join('');
  $$('[data-use-model]').forEach(button => button.addEventListener('click', () => { if (state.busy) { toast('识别完成后可切换模型。'); return; } switchModel(button.dataset.useModel); changeView('recognize'); }));
  icons();
}

function renderData() {
  const row = experiment();
  if (!row) return;
  const cfg = row.config, prep = cfg.preprocess;
  const items = [ ['数据来源', row.source], ['类别数量', `${cfg.classes.length} 类`], ['本地数据', row.data_available ? '数据目录可用' : '数据目录未提供'], ['训练 / 验证 / 测试', Object.values(cfg.split_sizes).map(value => value.toLocaleString()).join(' / ')], ['脉冲长度', `${prep.n_samples} 点`], ['采样率', `${prep.sample_rate_hz / 1e6} MHz`], ['FFT / 窗长 / 步长', `${prep.n_fft} / ${prep.win_len} / ${prep.hop_len}`], ['图像尺寸', `${prep.image_size} × ${prep.image_size}`], ['动态范围', `${prep.dyn_range_db} dB`], ['学习率', String(cfg.lr)], ['训练轮次', `${row.history.length} 轮`], ['论文特征预处理', cfg.model_kwargs.use_paper_preprocess ? '开启 · 6 次迭代 / 方差' : '关闭'] ];
  $('#data-details').innerHTML = items.map(([label, value]) => `<div class="data-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join('');
}

function renderHistory() {
  $('#clear-history').disabled = !state.records.length;
  if (!state.records.length) {
    $('#history-list').innerHTML = '<div class="history-empty"><i data-lucide="history"></i><p>暂无识别记录</p></div>';
  } else {
    $('#history-list').innerHTML = state.records.map((row, index) => `<div class="history-row"><div><strong>${escapeHtml(row.filename)}</strong><small>${new Date(row.created_at * 1000).toLocaleString('zh-CN')}</small></div><span>${escapeHtml(row.predictions[0].best_class)}</span><span>${percent(row.predictions[0].confidence)}%</span><span>${row.predictions.length} 条 / ${row.model === 'synthetic' ? '8 类' : '30 类'}</span><button class="icon-button" data-export-record="${index}" title="导出记录" aria-label="导出记录"><i data-lucide="download"></i></button></div>`).join('');
    $$('[data-export-record]').forEach(button => button.addEventListener('click', () => download(JSON.stringify(state.records[Number(button.dataset.exportRecord)], null, 2), 'aimc-prediction.json')));
  }
  icons();
}

function pickFile(file) {
  if (!file) return;
  if (!/\.(npy|npz|csv)$/i.test(file.name)) { showError('支持 .npy、.npz 和 .csv 文件。'); return; }
  if (file.size > 8 * 1024 * 1024) { showError('文件过大，上传上限为 8 MB。'); return; }
  state.file = file;
  $('#upload-name').textContent = file.name;
  $('#input-error').hidden = true;
}

function openImage(src, title) {
  if (!src) { toast('请先识别一条信号。'); return; }
  $('#dialog-image').src = src;
  $('#dialog-title').textContent = title;
  $('#image-dialog').showModal();
}

function bindEvents() {
  $$('.nav-item').forEach(button => button.addEventListener('click', () => changeView(button.dataset.view)));
  $$('.input-tab').forEach(button => button.addEventListener('click', () => setInput(button.dataset.inputTab)));
  $$('.signal-tab').forEach(button => button.addEventListener('click', () => setChart(button.dataset.signalTab)));
  $('#model-select').addEventListener('change', event => switchModel(event.target.value));
  $$('.experiment-select').forEach(select => select.addEventListener('change', event => switchModel(event.target.value)));
  $('#snr-slider').addEventListener('input', event => { $('#snr-output').textContent = `${Number(event.target.value) >= 0 ? '+' : ''}${event.target.value}`; });
  $('#confidence-slider').addEventListener('input', event => { $('#confidence-output').textContent = event.target.value; renderPrediction(); });
  $('#random-seed').addEventListener('click', () => { $('#seed-input').value = crypto.getRandomValues(new Uint32Array(1))[0]; });
  $('#run-button').addEventListener('click', run);
  $('#file-input').addEventListener('change', event => pickFile(event.target.files[0]));
  $('#dropzone').addEventListener('dragover', event => { event.preventDefault(); $('#dropzone').classList.add('dragging'); });
  $('#dropzone').addEventListener('dragleave', () => $('#dropzone').classList.remove('dragging'));
  $('#dropzone').addEventListener('drop', event => { event.preventDefault(); $('#dropzone').classList.remove('dragging'); pickFile(event.dataTransfer.files[0]); });
  $('#export-button').addEventListener('click', () => {
    let value = state.view === 'evaluation' ? experiment() : state.view === 'history' ? state.records : state.result;
    if (!value || (Array.isArray(value) && !value.length)) { toast('暂无可导出的结果。'); return; }
    if (state.view === 'recognize') { const { waveform, spectrogram, ...metadata } = value; value = metadata; }
    download(JSON.stringify(value, null, 2), `aimc-${state.view}.json`);
    toast('结果已导出');
  });
  $('#save-spectrum').addEventListener('click', () => {
    if (!state.result) { toast('请先识别一条信号。'); return; }
    download(state.result.spectrogram, 'aimc-spectrogram.png');
  });
  $('#expand-spectrum').addEventListener('click', () => openImage(state.result?.spectrogram, '功率时频图 · 首条脉冲'));
  $('#confusion-button').addEventListener('click', () => openImage(`/api/assets/${state.model}/confusion_matrix.png`, '独立测试 · 混淆矩阵'));
  $('#close-dialog').addEventListener('click', () => $('#image-dialog').close());
  $('#image-dialog').addEventListener('click', event => { if (event.target === $('#image-dialog')) { const rect = event.target.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.target.close(); } });
  $('#clear-history').addEventListener('click', () => { state.records = []; saveRecords(); renderHistory(); toast('识别记录已清空'); });
  window.addEventListener('resize', () => { if (state.view === 'evaluation') drawHistory(); if (state.chart === 'waveform' && state.result) drawWaveform(); });
}

async function initialize() {
  icons(); bindEvents(); saveRecords();
  try {
    const data = await api('/api/experiments');
    state.experiments = data.experiments;
    $$('.experiment-select').forEach(select => { select.innerHTML = data.experiments.map(row => `<option value="${row.id}">${escapeHtml(row.name)}</option>`).join(''); });
    updateOverview(); renderModels(); renderData(); renderEvaluation();
    await run();
  } catch (error) {
    showError(`服务连接失败：${error.message}`);
    $('#chart-loading').hidden = true;
    $('.connection').textContent = '服务未连接';
  }
}

initialize();
