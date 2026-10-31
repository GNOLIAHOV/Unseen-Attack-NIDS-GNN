import React, { useEffect, useMemo, useRef, useState } from "react";

export interface FlowPrediction {
  flow_index: number;
  src_node: string;
  dst_node: string;
  predicted_label: number;
  predicted_class: string;
  action: string;
  mean_prob: number;
  epistemic_std: number;
  topo_prob?: number;
  flow_prob?: number;
  pathway_discrepancy: number;
  osr_score: number;
  ground_truth_label?: number;
  ground_truth_attack?: string;
}

export interface PredictResponse {
  filename?: string;
  total_flows: number;
  mc_samples_T: number;
  std_threshold: number;
  latency_ms: number;
  summary_counts: {
    benign_0: number;
    known_attack_1: number;
    unseen_alert_2: number;
  };
  predictions: FlowPrediction[];
}

type ViewTab = "live" | "osr" | "models";
type Colourway = "graphite" | "paper";

const FALLBACK_OSR_METHODS = [
  {
    method: "1. Closed-Set Sigmoid Margin (T=1)",
    threshold: 0.5,
    auroc_open_set: 0.8227,
    aupr_unknown: 0.7918,
    oscr_score: 0.8151,
    known_class_ccr: 0.9262,
    unknown_flagged_as_open_set_rate: 0.1615,
    unknown_total_threat_recall: 0.8776,
  },
  {
    method: "2. Dual-Pathway Energy Discrepancy (T=1)",
    threshold: 0.15,
    auroc_open_set: 0.739,
    aupr_unknown: 0.7351,
    oscr_score: 0.7307,
    known_class_ccr: 0.913,
    unknown_flagged_as_open_set_rate: 0.1493,
    unknown_total_threat_recall: 0.8665,
  },
  {
    method: "3. Three-way Epistemic MC Dropout (T=20)",
    threshold: 0.15,
    auroc_open_set: 0.8531,
    aupr_unknown: 0.8045,
    oscr_score: 0.8417,
    known_class_ccr: 0.8595,
    unknown_flagged_as_open_set_rate: 0.4779,
    unknown_total_threat_recall: 0.918,
  },
  {
    method: "4. Hybrid Epistemic + Dual-Pathway OSR (T=20 - Proposed)",
    threshold: 0.15,
    auroc_open_set: 0.8531,
    aupr_unknown: 0.8044,
    oscr_score: 0.8417,
    known_class_ccr: 0.8594,
    unknown_flagged_as_open_set_rate: 0.4779,
    unknown_total_threat_recall: 0.9181,
  },
];

const MODEL_PROGRESSION = [
  {
    model: "1. B0: XGBoost (Tabular Baseline)",
    mechanism: "Gradient Boosted Trees trên 41 đặc trưng cạnh độc lập",
    seen_p: 99.14,
    seen_r: 98.77,
    seen_f1: 98.96,
    unseen_r: 2.13,
    bot_r: 0.0,
    infil_r: 4.76,
    detected: "5,536 / 259,458",
  },
  {
    model: "2. G0: Transductive GCN Baseline",
    mechanism: "2 lớp GCNConv + nn.Embedding(284339, 64) cố định",
    seen_p: 98.99,
    seen_r: 99.25,
    seen_f1: 99.12,
    unseen_r: 0.0,
    bot_r: 0.0,
    infil_r: 0.0,
    detected: "0 / 259,458",
  },
  {
    model: "3. G1: Inductive GINEConv",
    mechanism: "2 lớp GINEConv (Sum Aggregation) + Inductive Node Init 16 chiều",
    seen_p: 98.26,
    seen_r: 96.64,
    seen_f1: 97.44,
    unseen_r: 21.95,
    bot_r: 31.81,
    infil_r: 9.83,
    detected: "56,953 / 259,458",
  },
  {
    model: "4. G1_v2: Inductive Edge-Aware GATv2 (T=1)",
    mechanism: "2 lớp GATv2Conv (4 heads, edge_dim=41) + LayerNorm + Dual-Pathway",
    seen_p: 97.56,
    seen_r: 98.12,
    seen_f1: 97.84,
    unseen_r: 82.17,
    bot_r: 86.03,
    infil_r: 77.41,
    detected: "213,184 / 259,458",
  },
  {
    model: "5. GATv2 + Lọc Bất định 2 Nhánh (T=20)",
    mechanism: "Lọc báo động giả 2 nhánh truyền thống (tau_std = 0.15)",
    seen_p: 99.27,
    seen_r: 87.42,
    seen_f1: 92.97,
    unseen_r: 44.02,
    bot_r: 67.17,
    infil_r: 15.54,
    detected: "114,205 / 259,458",
  },
  {
    model: "6. GATv2 + Three-way OSR MC Dropout (T=20 - Đề xuất)",
    mechanism: "Quyết định 3 nhánh Tập Mở kết hợp Epistemic Std & Pathway Discrepancy",
    seen_p: 99.27,
    seen_r: 99.21,
    seen_f1: 99.24,
    unseen_r: 91.81,
    bot_r: 96.72,
    infil_r: 85.76,
    detected: "238,196 / 259,458",
  },
];

export default function App() {
  const params = new URLSearchParams(window.location.search);
  const initialTab = (params.get("view") as ViewTab) || "live";
  const initialColourway = (params.get("colourway") as Colourway) || "graphite";

  const [activeTab, setActiveTab] = useState<ViewTab>(
    ["live", "osr", "models"].includes(initialTab) ? initialTab : "live"
  );
  const [colourway, setColourway] = useState<Colourway>(
    initialColourway === "paper" ? "paper" : "graphite"
  );

  const [maxRows, setMaxRows] = useState<number>(400);
  const [mcSamples, setMcSamples] = useState<number>(20);
  const [stdThreshold, setStdThreshold] = useState<number>(0.15);

  const [loading, setLoading] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [batchData, setBatchData] = useState<PredictResponse | null>(null);
  const [metricsData, setMetricsData] = useState<Record<string, any>>({});

  const [selectedFlowIdx, setSelectedFlowIdx] = useState<number>(0);
  const [branchFilter, setBranchFilter] = useState<string>("all");
  const [attackFilter, setAttackFilter] = useState<string>("all");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-colourway", colourway);
    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.set("view", activeTab);
    nextUrl.searchParams.set("colourway", colourway);
    window.history.replaceState({}, "", nextUrl.toString());
  }, [activeTab, colourway]);

  const fetchMetrics = async () => {
    try {
      const res = await fetch("/metrics");
      if (res.ok) {
        const data = await res.json();
        setMetricsData(data.metrics || {});
      }
    } catch {
      // Keep fallback static metrics if offline
    }
  };

  const runDemoInference = async (rows = maxRows, T = mcSamples, tau = stdThreshold) => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const q = new URLSearchParams({
        max_rows: String(rows),
        mc_samples: String(T),
        std_threshold: String(tau),
      });
      const res = await fetch(`/predict_demo?${q.toString()}`, { method: "POST" });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Máy chủ trả về mã ${res.status}: ${text}`);
      }
      const data: PredictResponse = await res.json();
      setBatchData(data);
      if (data.predictions.length > 0) {
        const firstUnseen = data.predictions.find((p) => p.predicted_label === 2);
        setSelectedFlowIdx(firstUnseen ? firstUnseen.flow_index : data.predictions[0].flow_index);
      }
    } catch (err: any) {
      setErrorMsg(
        err?.message || "Không thể kết nối tới FastAPI Backend tại cổng 8000."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setErrorMsg(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const q = new URLSearchParams({
        max_rows: String(maxRows),
        mc_samples: String(mcSamples),
        std_threshold: String(stdThreshold),
      });
      const res = await fetch(`/predict_file?${q.toString()}`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Lỗi phân tích tệp (${res.status}): ${text}`);
      }
      const data: PredictResponse = await res.json();
      setBatchData(data);
      if (data.predictions.length > 0) {
        setSelectedFlowIdx(data.predictions[0].flow_index);
      }
    } catch (err: any) {
      setErrorMsg(err?.message || "Lỗi tải tệp lên máy chủ.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    runDemoInference(400, 20, 0.15);
  }, []);

  const predictions = batchData?.predictions || [];
  const selectedFlow = useMemo(
    () =>
      predictions.find((p) => p.flow_index === selectedFlowIdx) ||
      predictions[0] ||
      null,
    [predictions, selectedFlowIdx]
  );

  const uniqueAttacks = useMemo(() => {
    const s = new Set<string>();
    for (const p of predictions) {
      if (p.ground_truth_attack) s.add(p.ground_truth_attack);
    }
    return Array.from(s);
  }, [predictions]);

  const filteredFlows = useMemo(() => {
    return predictions.filter((p) => {
      if (branchFilter !== "all" && String(p.predicted_label) !== branchFilter) {
        return false;
      }
      if (attackFilter !== "all" && p.ground_truth_attack !== attackFilter) {
        return false;
      }
      return true;
    });
  }, [predictions, branchFilter, attackFilter]);

  const batchValidation = useMemo(() => {
    if (!predictions.length || predictions[0].ground_truth_label === undefined) {
      return null;
    }
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let unseenTotal = 0;
    let unseenCaught = 0;
    let unseenOsr2 = 0;

    for (const p of predictions) {
      const gt = p.ground_truth_label ?? 0;
      const predThreat = p.predicted_label >= 1 ? 1 : 0;
      if (gt === 1 && predThreat === 1) tp++;
      if (gt === 0 && predThreat === 1) fp++;
      if (gt === 1 && predThreat === 0) fn++;

      if (
        p.ground_truth_attack === "Bot" ||
        p.ground_truth_attack === "Infilteration"
      ) {
        unseenTotal++;
        if (p.predicted_label >= 1) unseenCaught++;
        if (p.predicted_label === 2) unseenOsr2++;
      }
    }

    const precision = tp / Math.max(tp + fp, 1);
    const recall = tp / Math.max(tp + fn, 1);
    const f1 = (2 * precision * recall) / Math.max(precision + recall, 1e-9);
    const unseenRecall = unseenCaught / Math.max(unseenTotal, 1);
    const unseenOsrRate = unseenOsr2 / Math.max(unseenTotal, 1);

    return {
      precision: (precision * 100).toFixed(2),
      recall: (recall * 100).toFixed(2),
      f1: (f1 * 100).toFixed(2),
      unseenRecall: (unseenRecall * 100).toFixed(2),
      unseenOsrRate: (unseenOsrRate * 100).toFixed(2),
      unseenTotal,
    };
  }, [predictions]);

  const osrSummary = metricsData.osr_open_set_metrics || null;
  const osrMethods = osrSummary?.osr_method_comparison || FALLBACK_OSR_METHODS;
  const osrProp = osrSummary?.proposed_hybrid_osr_summary || {
    auroc_open_set: 0.8531,
    aupr_unknown: 0.8044,
    oscr_score: 0.8417,
    overall_threat_precision: 0.9702,
    overall_threat_recall: 0.9435,
    overall_threat_f1: 0.9566,
    confusion_matrix_3x3: {
      matrix: [
        [55651, 859, 10527],
        [924, 116236, 15803],
        [21262, 114204, 123992],
      ],
    },
    unknown_family_breakdown: {
      Bot: {
        total_flows: 143097,
        predicted_class_0_benign_missed: 4698,
        predicted_class_1_known_attack: 96120,
        predicted_class_2_open_set_unknown: 42279,
        open_set_unknown_rate: 0.2955,
        total_intercept_recall: 0.9672,
        mean_epistemic_std: 0.1142,
        mean_pathway_discrepancy: 0.033,
      },
      Infilteration: {
        total_flows: 116361,
        predicted_class_0_benign_missed: 16564,
        predicted_class_1_known_attack: 18084,
        predicted_class_2_open_set_unknown: 81713,
        open_set_unknown_rate: 0.7022,
        total_intercept_recall: 0.8576,
        mean_epistemic_std: 0.1808,
        mean_pathway_discrepancy: 0.2283,
      },
    },
  };

  const unseenCount = batchData?.summary_counts.unseen_alert_2 ?? 0;
  const totalFlows = batchData?.total_flows || 0;
  const latencyMs = batchData?.latency_ms || 0;
  const throughputFps =
    latencyMs > 0 ? Math.round(totalFlows / (latencyMs / 1000)) : 0;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand-group">
            <a href="/" className="wordmark" aria-label="Trang chủ NF-UQ-NIDS-v2">
              NF-UQ-NIDS-v2 Console
            </a>
            <span className="brand-sub">
              Inductive Edge-Aware GATv2 và Nhận dạng Tập Mở Ba Nhánh (Nhóm 23)
            </span>
          </div>

          <nav className="nav-tabs" aria-label="Chế độ xem phân tích">
            <button
              type="button"
              className={`tab-btn ${activeTab === "live" ? "is-active" : ""}`}
              onClick={() => setActiveTab("live")}
            >
              1. Giám sát Luồng
            </button>
            <button
              type="button"
              className={`tab-btn ${activeTab === "osr" ? "is-active" : ""}`}
              onClick={() => setActiveTab("osr")}
            >
              2. Nhận dạng Tập Mở
            </button>
            <button
              type="button"
              className={`tab-btn ${activeTab === "models" ? "is-active" : ""}`}
              onClick={() => setActiveTab("models")}
            >
              3. Tiến trình Mô hình
            </button>
          </nav>

          <div className="control-cluster">
            <button
              type="button"
              className="btn"
              onClick={() =>
                setColourway(colourway === "graphite" ? "paper" : "graphite")
              }
            >
              Bề mặt: {colourway === "graphite" ? "Graphite" : "Paper"}
            </button>
          </div>
        </div>
      </header>

      {activeTab === "live" && (
        <section className="toolbar" aria-label="Tham số lấy mẫu MC Dropout">
          <div className="toolbar-inner">
            <div className="control-cluster">
              <label className="field-inline" htmlFor="input-max-rows">
                <span>Số luồng (Batch)</span>
                <input
                  id="input-max-rows"
                  name="max_rows"
                  type="number"
                  inputMode="numeric"
                  min={50}
                  max={1000}
                  step={50}
                  value={maxRows}
                  onChange={(e) => setMaxRows(Number(e.target.value))}
                  className="input-control"
                />
              </label>

              <label className="field-inline" htmlFor="input-mc-samples">
                <span>Mẫu MC (T)</span>
                <input
                  id="input-mc-samples"
                  name="mc_samples"
                  type="number"
                  inputMode="numeric"
                  min={2}
                  max={50}
                  step={2}
                  value={mcSamples}
                  onChange={(e) => setMcSamples(Number(e.target.value))}
                  className="input-control"
                />
              </label>

              <label className="field-inline" htmlFor="input-tau">
                <span>Ngưỡng OSR (tau)</span>
                <input
                  id="input-tau"
                  name="std_threshold"
                  type="number"
                  inputMode="decimal"
                  min={0.05}
                  max={0.35}
                  step={0.01}
                  value={stdThreshold}
                  onChange={(e) => setStdThreshold(Number(e.target.value))}
                  className="input-control"
                />
              </label>
            </div>

            <div className="control-cluster">
              <input
                ref={fileInputRef}
                type="file"
                name="flow_file"
                accept=".parquet,.csv"
                onChange={handleFileUpload}
                aria-label="Tải tệp dữ liệu luồng mạng"
                hidden
              />
              <button
                type="button"
                className="btn"
                disabled={loading}
                onClick={() => fileInputRef.current?.click()}
              >
                Tải tệp .parquet / .csv
              </button>
              <button
                type="button"
                className="btn btn--primary"
                disabled={loading}
                onClick={() => runDemoInference(maxRows, mcSamples, stdThreshold)}
              >
                {loading
                  ? "Đang suy luận đồ thị…"
                  : "Chạy gói Demo (Seen + Unseen)"}
              </button>
            </div>
          </div>
        </section>
      )}

      <main className="main-workspace">
        {activeTab === "live" && (
          <>
            {errorMsg && (
              <div className="state-panel" role="alert">
                <h2>Không thể thực thi suy luận luồng mạng</h2>
                <p>{errorMsg}</p>
                <div>
                  <button
                    type="button"
                    className="btn btn--primary"
                    onClick={() =>
                      runDemoInference(maxRows, mcSamples, stdThreshold)
                    }
                  >
                    Thử kết nối lại máy chủ
                  </button>
                </div>
              </div>
            )}

            {loading && !batchData && (
              <div className="state-panel" role="status">
                <h2>Đang dựng đồ thị con cục bộ và lấy mẫu MC Dropout…</h2>
                <p>
                  Hệ thống đang mã hóa cấu trúc láng giềng 2 bước (GATv2Conv) và
                  chạy {mcSamples} lượt truyền thuận ngẫu nhiên.
                </p>
              </div>
            )}

            {!loading && !errorMsg && predictions.length === 0 && (
              <div className="state-panel">
                <h2>Chưa có luồng mạng nào trong phiên làm việc</h2>
                <p>
                  Bấm nút chạy gói dữ liệu mẫu hoặc tải lên một tệp Parquet/CSV
                  để bắt đầu định lượng bất định.
                </p>
                <div>
                  <button
                    type="button"
                    className="btn btn--primary"
                    onClick={() =>
                      runDemoInference(maxRows, mcSamples, stdThreshold)
                    }
                  >
                    Nạp gói dữ liệu mẫu
                  </button>
                </div>
              </div>
            )}

            {batchData && predictions.length > 0 && (
              <>
                <section
                  className="panels panels--stats-5"
                  aria-label="Số đo tổng hợp trên lô dữ liệu"
                >
                  <div className="stat-cell">
                    <span className="stat-label">
                      Tổng số luồng mạng đã đo
                    </span>
                    <div className="reading reading--hero">
                      {totalFlows.toLocaleString("en-US")}
                    </div>
                    <span className="stat-note">
                      Nguồn: {batchData.filename || "demo_microbatch.parquet"}
                    </span>
                  </div>

                  <div className="stat-cell">
                    <span className="stat-label">
                      Nhánh 0: Known Benign (Allow)
                    </span>
                    <div className="reading reading--ok">
                      {batchData.summary_counts.benign_0.toLocaleString("en-US")}
                    </div>
                    <span className="stat-note measured">
                      {(
                        (batchData.summary_counts.benign_0 /
                          Math.max(totalFlows, 1)) *
                        100
                      ).toFixed(1)}
                      % tổng lưu lượng
                    </span>
                  </div>

                  <div className="stat-cell">
                    <span className="stat-label">
                      Nhánh 1: Known Attack (Block)
                    </span>
                    <div className="reading reading--cool">
                      {batchData.summary_counts.known_attack_1.toLocaleString(
                        "en-US"
                      )}
                    </div>
                    <span className="stat-note measured">
                      Mean Prob &gt;= 0.50, OSR &lt;= {batchData.std_threshold}
                    </span>
                  </div>

                  <div className="stat-cell">
                    <span className="stat-label">
                      Nhánh 2: Unseen / Open-Set Alert
                    </span>
                    <div className="reading reading--amber">
                      {unseenCount.toLocaleString("en-US")}
                    </div>
                    <span className="stat-note measured">
                      OSR Score &gt; {batchData.std_threshold} (T=
                      {batchData.mc_samples_T})
                    </span>
                  </div>

                  <div className="stat-cell">
                    <span className="stat-label">
                      Thời gian & Thông lượng lô
                    </span>
                    <div className="reading">
                      {latencyMs.toFixed(1)} ms
                    </div>
                    <span className="stat-note measured">
                      {throughputFps.toLocaleString("en-US")} flows/s
                    </span>
                  </div>
                </section>

                {batchValidation && (
                  <section
                    className="panels panels--stats-4"
                    aria-label="Kiểm chứng với nhãn thực tế trong gói mẫu"
                  >
                    <div className="stat-cell">
                      <span className="stat-label">
                        Precision phòng thủ thực tế (Batch)
                      </span>
                      <div className="reading reading--cool">
                        {batchValidation.precision}%
                      </div>
                      <span className="stat-note">
                        Đối chiếu với cột ground_truth_label
                      </span>
                    </div>
                    <div className="stat-cell">
                      <span className="stat-label">
                        Recall phòng thủ tổng thể (Batch)
                      </span>
                      <div className="reading reading--ok">
                        {batchValidation.recall}%
                      </div>
                      <span className="stat-note measured">
                        Threat F1: {batchValidation.f1}%
                      </span>
                    </div>
                    <div className="stat-cell">
                      <span className="stat-label">
                        Tỷ lệ chặn Unseen Attack (Bot & Infilteration)
                      </span>
                      <div className="reading reading--amber">
                        {batchValidation.unseenRecall}%
                      </div>
                      <span className="stat-note measured">
                        Trên {batchValidation.unseenTotal} luồng Unseen trong lô
                      </span>
                    </div>
                    <div className="stat-cell">
                      <span className="stat-label">
                        Tỷ lệ gắn cờ Nhánh 2 (Open-Set Alert)
                      </span>
                      <div className="reading">
                        {batchValidation.unseenOsrRate}%
                      </div>
                      <span className="stat-note">
                        Phát hiện trực tiếp qua ngưỡng bất định
                      </span>
                    </div>
                  </section>
                )}

                <section
                  className="panels panels--split"
                  aria-label="Không gian quyết định 2 chiều và thanh kiểm tra luồng mạng"
                >
                  <div>
                    <div className="panel-head">
                      <div>
                        <h2>
                          Không gian Quyết định Nhận thức: Mean Probability và
                          Epistemic Std
                        </h2>
                        <p className="panel-sub">
                          Bấm vào một điểm đo trên đồ thị để kiểm tra đường truyền
                          hai nhánh của luồng mạng đó.
                        </p>
                      </div>
                      <span className="chip chip--amber measured">
                        tau = {batchData.std_threshold.toFixed(2)}
                      </span>
                    </div>

                    <svg
                      viewBox="0 0 640 310"
                      width="100%"
                      height="310"
                      className="chart-svg"
                      role="img"
                      aria-label="Biểu đồ phân tán xác suất dự đoán trung bình và độ lệch chuẩn nhận thức"
                    >
                      <line
                        x1="54"
                        y1="26"
                        x2="616"
                        y2="26"
                        stroke="var(--color-rule)"
                        strokeWidth="1"
                      />
                      <line
                        x1="54"
                        y1="104"
                        x2="616"
                        y2="104"
                        stroke="var(--color-rule)"
                        strokeWidth="1"
                      />
                      <line
                        x1="54"
                        y1="182"
                        x2="616"
                        y2="182"
                        stroke="var(--color-rule)"
                        strokeWidth="1"
                      />
                      <line
                        x1="54"
                        y1="260"
                        x2="616"
                        y2="260"
                        stroke="var(--color-rule-2)"
                        strokeWidth="1"
                      />

                      {/* Vertical decision boundary p = 0.50 */}
                      <line
                        x1={54 + 0.5 * 562}
                        y1="20"
                        x2={54 + 0.5 * 562}
                        y2="260"
                        stroke="var(--color-rule-2)"
                        strokeWidth="1"
                      />

                      {/* Horizontal OSR threshold line tau */}
                      {(() => {
                        const maxY = 0.35;
                        const tauY =
                          260 -
                          Math.min(batchData.std_threshold / maxY, 1) * 234;
                        return (
                          <>
                            <line
                              x1="54"
                              y1={tauY}
                              x2="616"
                              y2={tauY}
                              stroke="var(--color-accent)"
                              strokeWidth="1.5"
                            />
                            <text
                              x="612"
                              y={tauY - 6}
                              textAnchor="end"
                              fill="var(--color-accent)"
                              fontSize="11"
                              fontFamily="var(--font-outlier)"
                            >
                              tau = {batchData.std_threshold.toFixed(2)}
                            </text>
                          </>
                        );
                      })()}

                      {predictions.map((pt) => {
                        const cx = 54 + Math.min(Math.max(pt.mean_prob, 0), 1) * 562;
                        const cy =
                          260 -
                          Math.min(Math.max(pt.epistemic_std, 0) / 0.35, 1) *
                            234;
                        const isSel =
                          selectedFlow &&
                          selectedFlow.flow_index === pt.flow_index;
                        const fill =
                          pt.predicted_label === 2
                            ? "var(--color-accent)"
                            : pt.predicted_label === 1
                            ? "var(--color-cool)"
                            : "var(--color-ok)";
                        return (
                          <circle
                            key={pt.flow_index}
                            cx={cx}
                            cy={cy}
                            r={isSel ? 6.5 : 3.8}
                            fill={fill}
                            fillOpacity={isSel ? 1 : 0.78}
                            stroke={isSel ? "var(--color-ink)" : "none"}
                            strokeWidth={isSel ? 1.5 : 0}
                            style={{ cursor: "pointer" }}
                            onClick={() => setSelectedFlowIdx(pt.flow_index)}
                          />
                        );
                      })}

                      <text
                        x="54"
                        y="282"
                        fill="var(--color-muted)"
                        fontSize="11"
                        fontFamily="var(--font-outlier)"
                      >
                        0.00
                      </text>
                      <text
                        x="335"
                        y="282"
                        textAnchor="middle"
                        fill="var(--color-muted)"
                        fontSize="11"
                        fontFamily="var(--font-outlier)"
                      >
                        0.50 (Ranh giới lớp)
                      </text>
                      <text
                        x="616"
                        y="282"
                        textAnchor="end"
                        fill="var(--color-muted)"
                        fontSize="11"
                        fontFamily="var(--font-outlier)"
                      >
                        1.00
                      </text>
                      <text
                        x="335"
                        y="302"
                        textAnchor="middle"
                        fill="var(--color-muted)"
                        fontSize="12"
                      >
                        Xác suất dự đoán trung bình qua T lượt MC (Mean Prob)
                      </text>

                      <text
                        x="46"
                        y="260"
                        textAnchor="end"
                        fill="var(--color-muted)"
                        fontSize="11"
                        fontFamily="var(--font-outlier)"
                      >
                        0.00
                      </text>
                      <text
                        x="46"
                        y="182"
                        textAnchor="end"
                        fill="var(--color-muted)"
                        fontSize="11"
                        fontFamily="var(--font-outlier)"
                      >
                        0.12
                      </text>
                      <text
                        x="46"
                        y="104"
                        textAnchor="end"
                        fill="var(--color-muted)"
                        fontSize="11"
                        fontFamily="var(--font-outlier)"
                      >
                        0.23
                      </text>
                      <text
                        x="46"
                        y="30"
                        textAnchor="end"
                        fill="var(--color-muted)"
                        fontSize="11"
                        fontFamily="var(--font-outlier)"
                      >
                        0.35
                      </text>
                    </svg>
                  </div>

                  <div>
                    <div className="panel-head">
                      <div>
                        <h2>Hồ sơ Đo lường Chi tiết Luồng Mạng</h2>
                        <p className="panel-sub">
                          Phân rã tín hiệu hai nhánh GATv2 và điểm Tập Mở
                        </p>
                      </div>
                      {selectedFlow && (
                        <span
                          className={`chip ${
                            selectedFlow.predicted_label === 2
                              ? "chip--amber"
                              : selectedFlow.predicted_label === 1
                              ? "chip--cool"
                              : "chip--ok"
                          }`}
                        >
                          {selectedFlow.action}
                        </span>
                      )}
                    </div>

                    {selectedFlow && (
                      <div className="audit-trail">
                        <div className="audit-entry">
                          <span className="stat-label">
                            Định danh luồng & Cặp nút giao tiếp
                          </span>
                          <div className="measured">
                            Flow #{selectedFlow.flow_index} (Node{" "}
                            {selectedFlow.src_node} đến Node{" "}
                            {selectedFlow.dst_node})
                          </div>
                        </div>

                        <div className="audit-entry">
                          <span className="stat-label">
                            Phân nhánh Tập Mở (Three-way Decision)
                          </span>
                          <div>
                            <strong>{selectedFlow.predicted_class}</strong>
                            {selectedFlow.ground_truth_attack && (
                              <span className="stat-note">
                                {" "}
                                (Nhãn thực tế: {selectedFlow.ground_truth_attack})
                              </span>
                            )}
                          </div>
                        </div>

                        <div className="bar-list">
                          <div className="bar-item">
                            <div className="bar-meta">
                              <span>Xác suất trung bình (Mean Prob)</span>
                              <span className="measured">
                                {selectedFlow.mean_prob.toFixed(4)}
                              </span>
                            </div>
                            <div className="bar-track">
                              <div
                                className="bar-fill"
                                style={{
                                  width: `${Math.min(
                                    selectedFlow.mean_prob * 100,
                                    100
                                  )}%`,
                                }}
                              />
                            </div>
                          </div>

                          <div className="bar-item">
                            <div className="bar-meta">
                              <span>
                                Độ lệch chuẩn nhận thức (Epistemic Std)
                              </span>
                              <span className="measured">
                                {selectedFlow.epistemic_std.toFixed(4)}
                              </span>
                            </div>
                            <div className="bar-track">
                              <div
                                className="bar-fill bar-fill--accent"
                                style={{
                                  width: `${Math.min(
                                    (selectedFlow.epistemic_std / 0.35) * 100,
                                    100
                                  )}%`,
                                }}
                              />
                            </div>
                          </div>

                          <div className="bar-item">
                            <div className="bar-meta">
                              <span>
                                Độ bất đồng hai nhánh (Pathway Discrepancy)
                              </span>
                              <span className="measured">
                                {selectedFlow.pathway_discrepancy.toFixed(4)}
                                {selectedFlow.topo_prob !== undefined &&
                                  selectedFlow.flow_prob !== undefined &&
                                  ` (p_topo=${selectedFlow.topo_prob.toFixed(
                                    3
                                  )}, p_flow=${selectedFlow.flow_prob.toFixed(
                                    3
                                  )})`}
                              </span>
                            </div>
                            <div className="bar-track">
                              <div
                                className="bar-fill bar-fill--muted"
                                style={{
                                  width: `${Math.min(
                                    selectedFlow.pathway_discrepancy * 100,
                                    100
                                  )}%`,
                                }}
                              />
                            </div>
                          </div>

                          <div className="bar-item">
                            <div className="bar-meta">
                              <span>
                                Điểm Nhận dạng Tập Mở (Hybrid OSR Score)
                              </span>
                              <span className="measured">
                                {selectedFlow.osr_score.toFixed(4)} (tau ={" "}
                                {batchData.std_threshold.toFixed(2)})
                              </span>
                            </div>
                            <div className="bar-track">
                              <div
                                className="bar-fill bar-fill--accent"
                                style={{
                                  width: `${Math.min(
                                    (selectedFlow.osr_score / 0.35) * 100,
                                    100
                                  )}%`,
                                }}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </section>

                <section
                  className="ledger-section"
                  aria-label="Sổ cái tra cứu từng luồng mạng"
                >
                  <div className="ledger-controls">
                    <div>
                      <h2>
                        Sổ cái Phân loại Luồng Mạng ({filteredFlows.length} /{" "}
                        {predictions.length} luồng)
                      </h2>
                    </div>

                    <div className="control-cluster">
                      <label className="field-inline" htmlFor="filter-branch">
                        <span>Nhánh quyết định</span>
                        <select
                          id="filter-branch"
                          name="branch_filter"
                          className="select-control"
                          value={branchFilter}
                          onChange={(e) => setBranchFilter(e.target.value)}
                        >
                          <option value="all">Tất cả nhánh (0, 1, 2)</option>
                          <option value="2">
                            Nhánh 2: Unseen / Open-Set Alert
                          </option>
                          <option value="1">Nhánh 1: Known Attack</option>
                          <option value="0">Nhánh 0: Known Benign</option>
                        </select>
                      </label>

                      {uniqueAttacks.length > 0 && (
                        <label className="field-inline" htmlFor="filter-attack">
                          <span>Nhãn thực tế</span>
                          <select
                            id="filter-attack"
                            name="attack_filter"
                            className="select-control"
                            value={attackFilter}
                            onChange={(e) => setAttackFilter(e.target.value)}
                          >
                            <option value="all">Tất cả họ lưu lượng</option>
                            {uniqueAttacks.map((atk) => (
                              <option key={atk} value={atk}>
                                {atk}
                              </option>
                            ))}
                          </select>
                        </label>
                      )}
                    </div>
                  </div>

                  <div className="table-scroll">
                    <table className="ledger-table">
                      <thead>
                        <tr>
                          <th>Chỉ số luồng</th>
                          <th>Nút nguồn</th>
                          <th>Nút đích</th>
                          <th>Mean Prob</th>
                          <th>Epistemic Std</th>
                          <th>Pathway Diff</th>
                          <th>OSR Score</th>
                          <th>Nhánh dự đoán</th>
                          <th>Hành động</th>
                          <th>Nhãn thực tế</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredFlows.slice(0, 250).map((r) => {
                          const isSelected =
                            selectedFlow &&
                            selectedFlow.flow_index === r.flow_index;
                          return (
                            <tr
                              key={r.flow_index}
                              className={`is-interactive ${
                                isSelected ? "is-selected" : ""
                              }`}
                              onClick={() => setSelectedFlowIdx(r.flow_index)}
                            >
                              <td className="measured">#{r.flow_index}</td>
                              <td className="measured">{r.src_node}</td>
                              <td className="measured">{r.dst_node}</td>
                              <td className="measured">
                                {r.mean_prob.toFixed(4)}
                              </td>
                              <td className="measured">
                                {r.epistemic_std.toFixed(4)}
                              </td>
                              <td className="measured">
                                {r.pathway_discrepancy.toFixed(4)}
                              </td>
                              <td className="measured">
                                {r.osr_score.toFixed(4)}
                              </td>
                              <td>
                                <span
                                  className={`chip ${
                                    r.predicted_label === 2
                                      ? "chip--amber"
                                      : r.predicted_label === 1
                                      ? "chip--cool"
                                      : "chip--ok"
                                  }`}
                                >
                                  {r.predicted_class}
                                </span>
                              </td>
                              <td className="measured">{r.action}</td>
                              <td>{r.ground_truth_attack || "N/A"}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </section>
              </>
            )}
          </>
        )}

        {activeTab === "osr" && (
          <>
            <section
              className="panels panels--stats-4"
              aria-label="Chỉ số chuẩn học thuật Open-Set Recognition"
            >
              <div className="stat-cell">
                <span className="stat-label">
                  AUROC (Known vs Open-Set Unknown)
                </span>
                <div className="reading reading--hero reading--cool">
                  {Number(osrProp.auroc_open_set).toFixed(4)}
                </div>
                <span className="stat-note">
                  Trên 459,458 luồng (200k Known + 259.4k Unseen)
                </span>
              </div>

              <div className="stat-cell">
                <span className="stat-label">
                  OSCR (Open-Set Classification Rate)
                </span>
                <div className="reading reading--hero reading--amber">
                  {Number(osrProp.oscr_score).toFixed(4)}
                </div>
                <span className="stat-note">
                  Diện tích đường cong CCR theo FPR_unknown
                </span>
              </div>

              <div className="stat-cell">
                <span className="stat-label">AUPR-Unknown (Open-Set)</span>
                <div className="reading reading--hero">
                  {Number(osrProp.aupr_unknown).toFixed(4)}
                </div>
                <span className="stat-note">
                  Độ chính xác xếp hạng mẫu ngoại lai
                </span>
              </div>

              <div className="stat-cell">
                <span className="stat-label">
                  Hiệu năng Phòng thủ Tập Mở (Threat F1)
                </span>
                <div className="reading reading--hero reading--ok">
                  {(Number(osrProp.overall_threat_f1) * 100).toFixed(2)}%
                </div>
                <span className="stat-note measured">
                  Precision:{" "}
                  {(Number(osrProp.overall_threat_precision) * 100).toFixed(2)}%
                  , Recall:{" "}
                  {(Number(osrProp.overall_threat_recall) * 100).toFixed(2)}%
                </span>
              </div>
            </section>

            <section
              className="panels panels--equal"
              aria-label="So sánh phương pháp OSR và ma trận nhầm lẫn 3 lớp"
            >
              <div>
                <div className="panel-head">
                  <div>
                    <h2>
                      Đối sánh 4 Phương pháp Nhận dạng Tập Mở (OSR Benchmark)
                    </h2>
                    <p className="panel-sub">
                      So sánh chỉ số OSCR và Tổng tỷ lệ chặn Unseen Attack
                    </p>
                  </div>
                </div>

                <div className="bar-list">
                  {osrMethods.map((m: any) => (
                    <div key={m.method} className="bar-item">
                      <div className="bar-meta">
                        <span>{m.method}</span>
                        <span className="measured">
                          AUROC {Number(m.auroc_open_set).toFixed(4)} | OSCR{" "}
                          {Number(m.oscr_score).toFixed(4)} | Recall{" "}
                          {(
                            Number(m.unknown_total_threat_recall) * 100
                          ).toFixed(2)}
                          %
                        </span>
                      </div>
                      <div className="bar-track">
                        <div
                          className={`bar-fill ${
                            m.method.includes("4.") ? "bar-fill--accent" : ""
                          }`}
                          style={{
                            width: `${
                              Number(m.unknown_total_threat_recall) * 100
                            }%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <div className="panel-head">
                  <div>
                    <h2>
                      Ma trận Nhầm lẫn Tập Mở 3 Lớp (3x3 Open-Set Matrix)
                    </h2>
                    <p className="panel-sub">
                      Không gian nhãn: 0 (Known Benign), 1 (Known Attack), 2
                      (Unseen Attack)
                    </p>
                  </div>
                </div>

                {(() => {
                  const mat = osrProp.confusion_matrix_3x3?.matrix || [
                    [55651, 859, 10527],
                    [924, 116236, 15803],
                    [21262, 114204, 123992],
                  ];
                  const rowNames = [
                    "Thực tế 0: Known Benign",
                    "Thực tế 1: Known Attack",
                    "Thực tế 2: Unseen Attack",
                  ];
                  return (
                    <div className="cm-grid">
                      <div className="cm-cell">
                        <span className="stat-label">Nhãn Thực / Dự đoán</span>
                      </div>
                      <div className="cm-cell">
                        <span className="stat-label">Dự đoán 0: Benign</span>
                      </div>
                      <div className="cm-cell">
                        <span className="stat-label">Dự đoán 1: Known Atk</span>
                      </div>
                      <div className="cm-cell">
                        <span className="stat-label">
                          Dự đoán 2: Open-Set Unk
                        </span>
                      </div>

                      {mat.map((row: number[], rIdx: number) => (
                        <React.Fragment key={rIdx}>
                          <div className="cm-cell">
                            <span className="stat-label">{rowNames[rIdx]}</span>
                          </div>
                          {row.map((val: number, cIdx: number) => (
                            <div
                              key={cIdx}
                              className={`cm-cell ${
                                rIdx === cIdx && rIdx < 2
                                  ? "cm-cell--diag"
                                  : rIdx === 2 && cIdx === 2
                                  ? "cm-cell--osr"
                                  : ""
                              }`}
                            >
                              <span className="reading">
                                {val.toLocaleString("en-US")}
                              </span>
                            </div>
                          ))}
                        </React.Fragment>
                      ))}
                    </div>
                  );
                })()}
              </div>
            </section>

            <section
              className="ledger-section"
              aria-label="Phân rã chi tiết từng họ tấn công chưa xuất hiện"
            >
              <div className="ledger-controls">
                <h2>
                  Phân rã Từng Họ Tấn công Chưa xuất hiện trong Huấn luyện (Bot
                  và Infilteration)
                </h2>
              </div>
              <div className="table-scroll">
                <table className="ledger-table">
                  <thead>
                    <tr>
                      <th>Họ Tấn công Unseen</th>
                      <th>Tổng số luồng</th>
                      <th>Bắt bởi Nhánh 1 (Known Attack)</th>
                      <th>Bắt bởi Nhánh 2 (Open-Set Alert)</th>
                      <th>Bỏ lọt vào Nhánh 0</th>
                      <th>Tỷ lệ gắn cờ Tập Mở (Nhánh 2)</th>
                      <th>Tổng Recall Phòng thủ (Nhánh 1 + 2)</th>
                      <th>Epistemic Std TB</th>
                      <th>Pathway Discrepancy TB</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(
                      osrProp.unknown_family_breakdown || {}
                    ).map(([atk, info]: [string, any]) => (
                      <tr key={atk}>
                        <td>
                          <strong>{atk}</strong>
                        </td>
                        <td className="measured">
                          {Number(info.total_flows).toLocaleString("en-US")}
                        </td>
                        <td className="measured">
                          {Number(
                            info.predicted_class_1_known_attack
                          ).toLocaleString("en-US")}
                        </td>
                        <td className="measured">
                          {Number(
                            info.predicted_class_2_open_set_unknown
                          ).toLocaleString("en-US")}
                        </td>
                        <td className="measured">
                          {Number(
                            info.predicted_class_0_benign_missed
                          ).toLocaleString("en-US")}
                        </td>
                        <td className="measured">
                          {(Number(info.open_set_unknown_rate) * 100).toFixed(
                            2
                          )}
                          %
                        </td>
                        <td className="measured">
                          <span className="chip chip--amber">
                            {(
                              Number(info.total_intercept_recall) * 100
                            ).toFixed(2)}
                            %
                          </span>
                        </td>
                        <td className="measured">
                          {Number(info.mean_epistemic_std).toFixed(4)}
                        </td>
                        <td className="measured">
                          {Number(info.mean_pathway_discrepancy).toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}

        {activeTab === "models" && (
          <>
            <section
              className="panels panels--split"
              aria-label="Tiến hóa hiệu năng qua 6 thế hệ mô hình và thông lượng hệ thống"
            >
              <div>
                <div className="panel-head">
                  <div>
                    <h2>
                      Tiến trình Cải tiến Khả năng Phát hiện Unseen Attack (B0
                      đến Three-way OSR)
                    </h2>
                    <p className="panel-sub">
                      Tỷ lệ phát hiện đúng trên 259,458 luồng Bot và
                      Infilteration
                    </p>
                  </div>
                </div>

                <div className="bar-list">
                  {MODEL_PROGRESSION.map((row) => (
                    <div key={row.model} className="bar-item">
                      <div className="bar-meta">
                        <span>{row.model}</span>
                        <span className="measured">
                          Test_seen F1: {row.seen_f1.toFixed(2)}% | Unseen
                          Recall: {row.unseen_r.toFixed(2)}% ({row.detected})
                        </span>
                      </div>
                      <div className="bar-track">
                        <div
                          className={`bar-fill ${
                            row.model.startsWith("6.") ? "bar-fill--accent" : ""
                          }`}
                          style={{
                            width: `${Math.max(row.unseen_r, 1)}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <div className="panel-head">
                  <div>
                    <h2>Đo lường Thông lượng & Độ trễ Thực thi</h2>
                    <p className="panel-sub">
                      Đánh giá trên 50 mini-batches liên tiếp (204,800 luồng
                      NetFlow, batch_size = 4,096)
                    </p>
                  </div>
                </div>

                <div className="audit-trail">
                  <div className="audit-entry">
                    <span className="stat-label">
                      Kịch bản 1: Suy luận Tất định Tiêu chuẩn (T = 1)
                    </span>
                    <div className="reading reading--cool">
                      168,625.47 flows/s
                    </div>
                    <span className="stat-note measured">
                      Độ trễ: 24.17 ms/batch (5.93 us/flow) | Peak RAM: 535.86
                      MB
                    </span>
                  </div>

                  <div className="audit-entry">
                    <span className="stat-label">
                      Kịch bản 2: Định lượng Bất định Tập Mở (MC Dropout T = 20)
                    </span>
                    <div className="reading reading--amber">
                      7,077.21 flows/s
                    </div>
                    <span className="stat-note measured">
                      Độ trễ: 578.65 ms/batch (141.30 us/flow) | Peak RAM:
                      540.41 MB
                    </span>
                  </div>

                  <div className="audit-entry">
                    <span className="stat-label">
                      Cơ chế tối ưu hóa mã hóa đồ thị con (Subgraph Caching)
                    </span>
                    <p className="panel-sub">
                      Bước truyền thông điệp GATv2Conv qua đồ thị láng giềng chỉ
                      thực hiện 1 lần duy nhất mỗi lô; 20 lượt lấy mẫu MC
                      Dropout chỉ chạy trên bộ giải mã hai nhánh nhẹ, giúp giữ
                      mức tiêu thụ bộ nhớ tăng thêm dưới 5 MB RAM.
                    </p>
                  </div>
                </div>
              </div>
            </section>

            <section
              className="ledger-section"
              aria-label="Bảng đối chiếu toàn diện 6 thế hệ mô hình"
            >
              <div className="ledger-controls">
                <h2>
                  Bảng Tổng hợp Kết quả Thực nghiệm Xuyên suốt Khóa luận
                </h2>
              </div>
              <div className="table-scroll">
                <table className="ledger-table">
                  <thead>
                    <tr>
                      <th>Thế hệ Mô hình / Cơ chế</th>
                      <th>Kiến trúc & Đặc trưng Node</th>
                      <th>Precision (Test_seen)</th>
                      <th>Recall (Test_seen)</th>
                      <th>F1-score (Test_seen)</th>
                      <th>Recall (Test_unseen)</th>
                      <th>Recall trên Bot</th>
                      <th>Recall trên Infilteration</th>
                      <th>Số mẫu Unseen bắt được</th>
                    </tr>
                  </thead>
                  <tbody>
                    {MODEL_PROGRESSION.map((r) => (
                      <tr key={r.model}>
                        <td>
                          <strong>{r.model}</strong>
                        </td>
                        <td>{r.mechanism}</td>
                        <td className="measured">{r.seen_p.toFixed(2)}%</td>
                        <td className="measured">{r.seen_r.toFixed(2)}%</td>
                        <td className="measured">{r.seen_f1.toFixed(2)}%</td>
                        <td className="measured">
                          <span
                            className={`chip ${
                              r.model.startsWith("6.")
                                ? "chip--amber"
                                : "chip--cool"
                            }`}
                          >
                            {r.unseen_r.toFixed(2)}%
                          </span>
                        </td>
                        <td className="measured">{r.bot_r.toFixed(2)}%</td>
                        <td className="measured">{r.infil_r.toFixed(2)}%</td>
                        <td className="measured">{r.detected}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}
      </main>

      <footer className="footer-bar">
        <span>
          Khóa luận Tốt nghiệp Nhóm 23: Nguyễn Đức Hưng Thịnh & Võ Hải Long
          (GVHD: TS. Võ Quang Hoàng Khang)
        </span>
        <span className="measured">
          Dataset: NF-UQ-NIDS-v2 (75,867,694 flows, 284,339 IP nodes)
        </span>
      </footer>
    </div>
  );
}
