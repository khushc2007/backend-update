import express, { Request, Response } from "express";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json());

/* ===============================
   CONFIG
================================ */
const BATCH_SIZE = 5;
const FLATLINE_WINDOW = 4;
const FLATLINE_EPSILON = 0.01;
const RECAL_DISAGREEMENT_THRESHOLD = 0.35;
const RECAL_CORRECTION_FACTOR = 0.88;
const MAX_FINGERPRINT_HISTORY = 20;

/* ===============================
   TYPES
================================ */
type Stage = "pre_lamella" | "post_lamella";

type SensorReading = {
  ph: number;
  turbidity: number;
  tds: number;
  timestamp: number;
  stage: Stage;
};

type SessionState = {
  active: boolean;
  completed: boolean;
  startedAt: number | null;
};

type SystemPhase =
  | "IDLE"
  | "COLLECTING"
  | "ANALYZED"
  | "TRANSFERRING_MAIN"
  | "POST_FILTRATION"
  | "COMPLETE";

type PumpCommand = "START_PUMP_A" | "START_PUMP_B" | "START_PUMP_C" | "STOP_ALL";
type Bracket = "F1" | "F2" | "F3" | "F4" | "F5";

type WQIResult = {
  score: number;
  stage: Stage;
  phContribution: number;
  turbidityContribution: number;
  tdsContribution: number;
  interpretation: "excellent" | "good" | "marginal" | "poor" | "reject";
};

type ConfidenceResult = {
  score: number;
  level: "high" | "medium" | "low" | "critical";
  phTurbidityAgreement: number;
  phTdsAgreement: number;
  turbidityTdsAgreement: number;
  recommendation: "proceed" | "extend_ec_cycle" | "re_run_cycle" | "discard";
  disagreementFlags: string[];
};

type FlatlineResult = {
  ph: boolean;
  turbidity: boolean;
  tds: boolean;
  anyFlatlined: boolean;
  failsafeTriggered: boolean;
  details: string[];
};

type RecalibrationResult = {
  triggered: boolean;
  reason: string | null;
  correctedTurbidity: number | null;
  originalTurbidity: number | null;
  disagreementScore: number;
};

type CycleFingerprint = {
  cycleId: string;
  capturedAt: number;
  durationMs: number;
  phSlope: number;
  turbiditySlope: number;
  tdsSlope: number;
  anomalyScore: number;
  anomalyFlags: string[];
};

type PredictionResult = {
  bracket: Bracket;
  reusable: boolean;
  suggestedTank: "A" | "B";
  filtrationMethod: string;
  decidedAt: number;
  wqi: WQIResult;
  confidence: ConfidenceResult;
  flatline: FlatlineResult;
  recalibration: RecalibrationResult;
  stageAware: { stage: Stage; note: string };
  cycleFingerprint: CycleFingerprint | null;
};

/* ===============================
   STATE
================================ */
let sessionReadings: SensorReading[] = [];
let session: SessionState = { active: false, completed: false, startedAt: null };
let systemPhase: SystemPhase = "IDLE";
let currentStage: Stage = "post_lamella";
let lastPrediction: PredictionResult | null = null;
let pendingPumpCommand: PumpCommand | null = null;
let commandDelivered = false;
let fingerprintHistory: (CycleFingerprint & { phCurve: number[]; turbidityCurve: number[]; tdsCurve: number[] })[] = [];

/* ===============================
   HELPERS
================================ */
function computeAverage(readings: SensorReading[]) {
  const sum = readings.reduce(
    (acc, r) => { acc.ph += r.ph; acc.turbidity += r.turbidity; acc.tds += r.tds; return acc; },
    { ph: 0, turbidity: 0, tds: 0 }
  );
  return {
    ph: +(sum.ph / readings.length).toFixed(2),
    turbidity: +(sum.turbidity / readings.length).toFixed(2),
    tds: Math.round(sum.tds / readings.length),
  };
}

function filtrationBracket(turbidity: number, tds: number): Bracket {
  if (tds > 1500) return "F5";
  if (tds >= 1000) return "F4";
  if (turbidity > 30) return "F3";
  if (turbidity > 10) return "F2";
  return "F1";
}

function filtrationMethod(bracket: string): string {
  const map: Record<string, string> = {
    F1: "Sediment + Carbon polishing",
    F2: "Sand + Carbon filtration",
    F3: "Coagulation + Sand filtration",
    F4: "Advanced treatment (discard)",
    F5: "RO / Disposal",
  };
  return map[bracket] ?? "Unknown";
}

/* ===============================
   ① COMPOSITE WQI — stage-aware weights
================================ */
function computeWQI(avg: { ph: number; turbidity: number; tds: number }, stage: Stage): WQIResult {
  const phScore = avg.ph >= 6.5 && avg.ph <= 8.0
    ? 100
    : Math.max(0, 100 - Math.abs(avg.ph - 7.25) * 25);
  const turbidityScore = Math.max(0, 100 - (avg.turbidity / 50) * 100);
  const tdsScore = avg.tds <= 500 ? 100 : avg.tds >= 2000 ? 0 : Math.max(0, 100 - ((avg.tds - 500) / 1500) * 100);

  // Key novelty: turbidity gets higher weight post-lamella (floc has settled = final reading)
  // TDS gets higher weight pre-lamella (leading contamination indicator before settling)
  const w = stage === "post_lamella"
    ? { ph: 0.25, turbidity: 0.50, tds: 0.25 }
    : { ph: 0.20, turbidity: 0.30, tds: 0.50 };

  const score = +(phScore * w.ph + turbidityScore * w.turbidity + tdsScore * w.tds).toFixed(1);
  const interpretation: WQIResult["interpretation"] =
    score >= 80 ? "excellent" : score >= 65 ? "good" : score >= 50 ? "marginal" : score >= 30 ? "poor" : "reject";

  return {
    score, stage,
    phContribution: +(phScore * w.ph).toFixed(1),
    turbidityContribution: +(turbidityScore * w.turbidity).toFixed(1),
    tdsContribution: +(tdsScore * w.tds).toFixed(1),
    interpretation,
  };
}

/* ===============================
   ② CONFIDENCE — cross-sensor agreement
================================ */
function computeConfidence(readings: SensorReading[]): ConfidenceResult {
  if (readings.length === 0) {
    return { score: 0, level: "critical", phTurbidityAgreement: 0, phTdsAgreement: 0, turbidityTdsAgreement: 0, recommendation: "discard", disagreementFlags: ["no_readings"] };
  }
  const avg = computeAverage(readings);
  const flags: string[] = [];

  const phP = avg.ph < 6.0 || avg.ph > 8.5 ? Math.min(1, Math.abs(avg.ph - 7.25) / 3) : 0;
  const turbP = Math.min(1, avg.turbidity / 50);
  const tdsP = Math.min(1, avg.tds / 2000);

  const ptA = +(1 - Math.abs(phP - turbP)).toFixed(3);
  const pdA = +(1 - Math.abs(phP - tdsP)).toFixed(3);
  const tdA = +(1 - Math.abs(turbP - tdsP)).toFixed(3);

  if (ptA < 0.65) flags.push("ph_turbidity_mismatch");
  if (pdA < 0.65) flags.push("ph_tds_mismatch");
  if (tdA < 0.65) flags.push("turbidity_tds_mismatch");

  const score = +(ptA * 0.35 + pdA * 0.30 + tdA * 0.35).toFixed(3);
  const level: ConfidenceResult["level"] = score >= 0.85 ? "high" : score >= 0.70 ? "medium" : score >= 0.50 ? "low" : "critical";
  const recommendation: ConfidenceResult["recommendation"] =
    score >= 0.85 ? "proceed" : score >= 0.70 ? "extend_ec_cycle" : score >= 0.50 ? "re_run_cycle" : "discard";

  return { score, level, phTurbidityAgreement: ptA, phTdsAgreement: pdA, turbidityTdsAgreement: tdA, recommendation, disagreementFlags: flags };
}

/* ===============================
   ③ FLATLINE DETECTION — sensor death failsafe
================================ */
function detectFlatlines(readings: SensorReading[]): FlatlineResult {
  const details: string[] = [];
  if (readings.length < FLATLINE_WINDOW) {
    return { ph: false, turbidity: false, tds: false, anyFlatlined: false, failsafeTriggered: false, details: ["insufficient_readings"] };
  }
  const recent = readings.slice(-FLATLINE_WINDOW);
  const check = (vals: number[], name: string): boolean => {
    const flat = (Math.max(...vals) - Math.min(...vals)) <= FLATLINE_EPSILON;
    if (flat) details.push(`${name}_flatlined_at_${vals[0].toFixed(3)}`);
    return flat;
  };
  const ph = check(recent.map(r => r.ph), "ph");
  const turbidity = check(recent.map(r => r.turbidity), "turbidity");
  const tds = check(recent.map(r => r.tds), "tds");
  const anyFlatlined = ph || turbidity || tds;
  if (anyFlatlined) details.push("failsafe_triggered_safe_discard_enforced");
  return { ph, turbidity, tds, anyFlatlined, failsafeTriggered: anyFlatlined, details };
}

/* ===============================
   ④ CROSS-SENSOR RECALIBRATION
   TDS + pH as ground truth validators for turbidity
================================ */
function attemptRecalibration(avg: { ph: number; turbidity: number; tds: number }): RecalibrationResult {
  const tdsSignal = avg.tds <= 300 ? 1.0 : avg.tds >= 1500 ? 0.0 : 1.0 - (avg.tds - 300) / 1200;
  const phSignal = avg.ph >= 6.5 && avg.ph <= 8.0 ? 1.0 : Math.max(0, 1.0 - Math.abs(avg.ph - 7.25) / 2.5);
  const groundTruth = tdsSignal * 0.6 + phSignal * 0.4;
  const turbiditySignal = Math.max(0, 1 - avg.turbidity / 50);
  const disagreementScore = +Math.abs(groundTruth - turbiditySignal).toFixed(3);

  if (disagreementScore < RECAL_DISAGREEMENT_THRESHOLD) {
    return { triggered: false, reason: null, correctedTurbidity: null, originalTurbidity: null, disagreementScore };
  }

  const directionClean = groundTruth > turbiditySignal;
  const correctedTurbidity = directionClean
    ? +(avg.turbidity * RECAL_CORRECTION_FACTOR).toFixed(2)
    : +(avg.turbidity / RECAL_CORRECTION_FACTOR).toFixed(2);

  return {
    triggered: true,
    reason: `turbidity_disagrees_with_tds_ph_ground_truth (score=${disagreementScore})`,
    correctedTurbidity,
    originalTurbidity: avg.turbidity,
    disagreementScore,
  };
}

/* ===============================
   ⑤ CYCLE FINGERPRINTING
   Capture sensor curve shape per cycle, detect anomalies vs history
================================ */
function buildFingerprint(readings: SensorReading[], startedAt: number) {
  const cycleId = `cycle_${Date.now()}`;
  const durationMs = Date.now() - startedAt;
  const phCurve = readings.map(r => r.ph);
  const turbidityCurve = readings.map(r => r.turbidity);
  const tdsCurve = readings.map(r => r.tds);

  const slope = (vals: number[]) => {
    if (vals.length < 2) return 0;
    const diffs = vals.slice(1).map((v, i) => v - vals[i]);
    return +(diffs.reduce((a, b) => a + b, 0) / diffs.length).toFixed(4);
  };

  const phSlope = slope(phCurve);
  const turbiditySlope = slope(turbidityCurve);
  const tdsSlope = slope(tdsCurve);

  const flags: string[] = [];
  let anomalyScore = 0;

  if (Math.abs(turbiditySlope) < 0.001 && turbidityCurve.length >= 3) { flags.push("turbidity_curve_too_flat"); anomalyScore += 0.35; }
  if (turbiditySlope > 0.5) { flags.push("turbidity_increasing_during_ec"); anomalyScore += 0.45; }
  if (Math.abs(tdsSlope) > 50) { flags.push("tds_spike_during_cycle"); anomalyScore += 0.30; }
  if (durationMs < 5000) { flags.push("cycle_too_short"); anomalyScore += 0.25; }

  if (fingerprintHistory.length >= 3) {
    const avgHistSlope = fingerprintHistory.slice(-5).reduce((a, f) => a + f.turbiditySlope, 0) / Math.min(5, fingerprintHistory.length);
    const dev = Math.abs(turbiditySlope - avgHistSlope);
    if (dev > 0.8) { flags.push(`turbidity_slope_deviation_${dev.toFixed(2)}`); anomalyScore += Math.min(0.4, dev * 0.2); }
  }

  return {
    cycleId, capturedAt: Date.now(), durationMs,
    phCurve, turbidityCurve, tdsCurve,
    phSlope, turbiditySlope, tdsSlope,
    anomalyScore: +Math.min(1, anomalyScore).toFixed(3),
    anomalyFlags: flags,
  };
}

/* ===============================
   ⑥ STAGE-AWARE NOTE
================================ */
function stageNote(stage: Stage, bracket: Bracket): string {
  if (stage === "post_lamella") {
    return bracket === "F1" || bracket === "F2"
      ? "Post-lamella: EC + settling effective. Clean routing confirmed."
      : "Post-lamella: residual contamination. EC may need extended cycle.";
  }
  return bracket === "F1" || bracket === "F2"
    ? "Pre-lamella: water clean before settling. Lamella pass safe."
    : "Pre-lamella: contamination load detected. Lamella settling required.";
}

/* ===============================
   ROUTES
================================ */
app.get("/", (_req, res) => { res.send("Water IQ Backend v2 — Edge Intelligence"); });

// ── INGEST ──
app.post("/ingest", (req: Request, res: Response) => {
  const { ph, turbidity, tds, stage } = req.body;
  if (typeof ph !== "number" || typeof turbidity !== "number" || typeof tds !== "number") {
    return res.status(400).json({ error: "Invalid sensor data" });
  }
  if (systemPhase === "IDLE") {
    systemPhase = "COLLECTING"; session.active = true; session.completed = false; session.startedAt = Date.now();
  }
  if (sessionReadings.length >= BATCH_SIZE) {
    return res.json({ status: "ignored", reason: "batch_complete", collected: sessionReadings.length, phase: systemPhase });
  }
  if (systemPhase === "COLLECTING") {
    const resolvedStage: Stage = stage === "pre_lamella" || stage === "post_lamella" ? stage : currentStage;
    sessionReadings.push({ ph, turbidity, tds, timestamp: Date.now(), stage: resolvedStage });
    const flatlineCheck = detectFlatlines(sessionReadings);
    if (flatlineCheck.failsafeTriggered) {
      return res.json({ status: "received_with_warning", collected: sessionReadings.length, phase: systemPhase, warning: "flatline_detected", flatline: flatlineCheck });
    }
  }
  res.json({ status: "received", collected: sessionReadings.length, phase: systemPhase });
});

// ── SESSION ──
app.post("/session/start", (req: Request, res: Response) => {
  const { stage } = req.body;
  sessionReadings = []; session.active = true; session.completed = false; session.startedAt = Date.now();
  systemPhase = "COLLECTING"; lastPrediction = null; pendingPumpCommand = null; commandDelivered = false;
  currentStage = stage === "pre_lamella" || stage === "post_lamella" ? stage : "post_lamella";
  res.json({ status: "session_started", batchSize: BATCH_SIZE, stage: currentStage });
});

app.post("/session/reset", (_req, res) => {
  session = { active: false, completed: false, startedAt: null };
  sessionReadings = []; systemPhase = "IDLE"; lastPrediction = null; pendingPumpCommand = null; commandDelivered = false;
  res.json({ status: "session_reset" });
});

app.get("/session/status", (_req, res) => {
  res.json({ active: session.active, completed: session.completed, collected: sessionReadings.length, phase: systemPhase, stage: currentStage });
});

app.get("/session/readings", (_req, res) => {
  res.json({ readings: sessionReadings });
});

// ── ANALYZE ──
app.post("/analyze-water", (_req, res) => {
  if (systemPhase !== "COLLECTING") return res.status(400).json({ error: "Analysis not allowed in current phase", phase: systemPhase });
  if (sessionReadings.length < BATCH_SIZE) return res.status(400).json({ error: "Insufficient data", required: BATCH_SIZE, current: sessionReadings.length });

  // 1. Flatline check — failsafe first
  const flatline = detectFlatlines(sessionReadings);
  if (flatline.failsafeTriggered) {
    session.active = false; session.completed = true; systemPhase = "ANALYZED";
    lastPrediction = {
      bracket: "F5", reusable: false, suggestedTank: "B",
      filtrationMethod: "SENSOR_FAILSAFE — discard enforced", decidedAt: Date.now(),
      wqi: { score: 0, stage: currentStage, phContribution: 0, turbidityContribution: 0, tdsContribution: 0, interpretation: "reject" },
      confidence: { score: 0, level: "critical", phTurbidityAgreement: 0, phTdsAgreement: 0, turbidityTdsAgreement: 0, recommendation: "discard", disagreementFlags: ["sensor_death"] },
      flatline,
      recalibration: { triggered: false, reason: "skipped_due_to_flatline", correctedTurbidity: null, originalTurbidity: null, disagreementScore: 0 },
      stageAware: { stage: currentStage, note: "Sensor failsafe — routing blocked." },
      cycleFingerprint: null,
    };
    sessionReadings = [];
    return res.json({ ...lastPrediction, failsafe: true });
  }

  // 2. Average
  const avg = computeAverage(sessionReadings);

  // 3. Recalibration
  const recalibration = attemptRecalibration(avg);
  const effTurbidity = recalibration.triggered && recalibration.correctedTurbidity !== null
    ? recalibration.correctedTurbidity : avg.turbidity;
  const effAvg = { ...avg, turbidity: effTurbidity };

  // 4. Bracket
  let bracket = filtrationBracket(effAvg.turbidity, effAvg.tds);

  // 5. WQI
  const wqi = computeWQI(effAvg, currentStage);

  // 6. Confidence — may override bracket
  const confidence = computeConfidence(sessionReadings);
  let reusable = bracket === "F1" || bracket === "F2";
  if (confidence.recommendation === "discard") { bracket = "F5"; reusable = false; }
  else if (confidence.recommendation === "re_run_cycle" && reusable) { bracket = "F3"; reusable = false; }

  // 7. Fingerprint
  const fp = buildFingerprint(sessionReadings, session.startedAt ?? Date.now());
  fingerprintHistory.push(fp);
  if (fingerprintHistory.length > MAX_FINGERPRINT_HISTORY) fingerprintHistory.shift();

  lastPrediction = {
    bracket, reusable, suggestedTank: reusable ? "A" : "B",
    filtrationMethod: filtrationMethod(bracket), decidedAt: Date.now(),
    wqi, confidence, flatline, recalibration,
    stageAware: { stage: currentStage, note: stageNote(currentStage, bracket) },
    cycleFingerprint: {
      cycleId: fp.cycleId, capturedAt: fp.capturedAt, durationMs: fp.durationMs,
      phSlope: fp.phSlope, turbiditySlope: fp.turbiditySlope, tdsSlope: fp.tdsSlope,
      anomalyScore: fp.anomalyScore, anomalyFlags: fp.anomalyFlags,
    },
  };

  session.active = false; session.completed = true; systemPhase = "ANALYZED";
  sessionReadings = [];
  res.json({ ...lastPrediction, average: effAvg, originalAverage: avg });
});

// ── PREDICTION ──
app.get("/prediction/latest", (_req, res) => {
  if (!lastPrediction) return res.status(404).json({ error: "No prediction available" });
  res.json(lastPrediction);
});

// ── PUMP COMMANDS ──
app.post("/pump/command", (req: Request, res: Response) => {
  const { command } = req.body;
  if (!["START_PUMP_A", "START_PUMP_B", "START_PUMP_C", "STOP_ALL"].includes(command)) return res.status(400).json({ error: "Invalid command" });
  if (systemPhase !== "ANALYZED" && command !== "STOP_ALL") return res.status(400).json({ error: "Invalid system phase" });
  pendingPumpCommand = command; commandDelivered = false;
  if (command === "STOP_ALL") systemPhase = "COMPLETE";
  else if (command === "START_PUMP_C") systemPhase = "POST_FILTRATION";
  else systemPhase = "TRANSFERRING_MAIN";
  res.json({ status: "command_queued", command });
});

app.get("/pump/command", (_req, res) => {
  if (!pendingPumpCommand || commandDelivered) return res.json({ command: null });
  commandDelivered = true;
  res.json({ command: pendingPumpCommand });
});

app.post("/pump/ack", (_req, res) => {
  pendingPumpCommand = null; commandDelivered = false; systemPhase = "IDLE";
  res.json({ status: "acknowledged" });
});

// ── DIAGNOSTICS (new) ──
app.get("/diagnostics", (_req, res) => {
  res.json({
    systemPhase, currentStage, session, lastPrediction,
    fingerprintHistoryCount: fingerprintHistory.length,
    latestFingerprint: fingerprintHistory.length > 0 ? fingerprintHistory[fingerprintHistory.length - 1] : null,
    anomalyTrend: fingerprintHistory.length >= 3
      ? +(fingerprintHistory.slice(-3).reduce((a, f) => a + f.anomalyScore, 0) / 3).toFixed(3)
      : null,
  });
});

// ── FINGERPRINT HISTORY (new) ──
app.get("/fingerprints", (_req, res) => {
  res.json({
    count: fingerprintHistory.length,
    fingerprints: fingerprintHistory.map(f => ({
      cycleId: f.cycleId, capturedAt: f.capturedAt, durationMs: f.durationMs,
      phSlope: f.phSlope, turbiditySlope: f.turbiditySlope, tdsSlope: f.tdsSlope,
      anomalyScore: f.anomalyScore, anomalyFlags: f.anomalyFlags,
    })),
  });
});

/* ===============================
   SERVER
================================ */
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => { console.log(`Water IQ Backend v2 running on port ${PORT}`); });
