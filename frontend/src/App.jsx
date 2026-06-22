import React, { useState, useEffect, useCallback } from 'react';
import {
  Stethoscope, Activity, FileText, ChevronRight,
  AlertCircle, ShieldCheck, HeartPulse, HelpCircle,
  FlaskConical, UserCircle, ChevronDown, ChevronUp
} from 'lucide-react';

/* ── Clinical Reference Cases ─────────────────────────────── */
const PRESETS = [
  {
    id: 'healthy',
    name: 'Standard Baseline',
    desc: 'Normal hepatic function',
    age: 35, gender: 'Male',
    total_bilirubin: 0.7, direct_bilirubin: 0.1,
    alkaline_phosphotase: 95, alamine_aminotransferase: 22,
    aspartate_aminotransferase: 20, total_protiens: 7.2,
    albumin: 4.2, albumin_and_globulin_ratio: 1.4,
  },
  {
    id: 'early',
    name: 'Hepatitis Pattern',
    desc: 'Elevated transaminases',
    age: 45, gender: 'Male',
    total_bilirubin: 2.1, direct_bilirubin: 0.8,
    alkaline_phosphotase: 220, alamine_aminotransferase: 72,
    aspartate_aminotransferase: 65, total_protiens: 6.8,
    albumin: 3.8, albumin_and_globulin_ratio: 1.1,
  },
  {
    id: 'cirrhosis',
    name: 'Advanced Cirrhosis',
    desc: 'Severe enzyme derangement',
    age: 62, gender: 'Male',
    total_bilirubin: 8.4, direct_bilirubin: 5.2,
    alkaline_phosphotase: 680, alamine_aminotransferase: 185,
    aspartate_aminotransferase: 210, total_protiens: 5.4,
    albumin: 2.4, albumin_and_globulin_ratio: 0.6,
  },
];

const FIELDS = [
  { key: 'total_bilirubin', label: 'Total Bilirubin', unit: 'mg/dL', step: 0.1 },
  { key: 'direct_bilirubin', label: 'Direct Bilirubin', unit: 'mg/dL', step: 0.1 },
  { key: 'alkaline_phosphotase', label: 'Alkaline Phosphatase', unit: 'IU/L', step: 1 },
  { key: 'alamine_aminotransferase', label: 'ALT (SGPT)', unit: 'U/L', step: 1 },
  { key: 'aspartate_aminotransferase', label: 'AST (SGOT)', unit: 'U/L', step: 1 },
  { key: 'total_protiens', label: 'Total Proteins', unit: 'g/dL', step: 0.1 },
  { key: 'albumin', label: 'Serum Albumin', unit: 'g/dL', step: 0.1 },
  { key: 'albumin_and_globulin_ratio', label: 'A/G Ratio', unit: 'ratio', step: 0.01},
];

const DEFAULTS = {
  age: 45, gender: 'Male',
  total_bilirubin: 1.0, direct_bilirubin: 0.3,
  alkaline_phosphotase: 200, alamine_aminotransferase: 35,
  aspartate_aminotransferase: 32, total_protiens: 6.8,
  albumin: 3.5, albumin_and_globulin_ratio: 1.0,
};

export default function App() {
  const [form, setForm] = useState(DEFAULTS);
  const [activePreset, setPreset] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [apiStatus, setStatus] = useState('loading');
  const [modelInfo, setModelInfo] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [timeline, setTimeline] = useState([]);

  useEffect(() => {
    fetch('/api/health')
      .then(r => r.json())
      .then(d => {
        setStatus(d.model_loaded ? 'ok' : 'error');
        setModelInfo(d);
      })
      .catch(() => setStatus('error'));
  }, []);

  const applyPreset = (p) => {
    const { id, name, desc, ...vals } = p;
    setForm(vals);
    setPreset(p.id);
  };

  const setField = (k, v) => {
    setForm(f => ({ ...f, [k]: v }));
    setPreset('');
  };

  const analyze = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...form,
          age: parseInt(form.age),
          total_bilirubin: parseFloat(form.total_bilirubin),
          direct_bilirubin: parseFloat(form.direct_bilirubin),
          alkaline_phosphotase: parseFloat(form.alkaline_phosphotase),
          alamine_aminotransferase: parseFloat(form.alamine_aminotransferase),
          aspartate_aminotransferase: parseFloat(form.aspartate_aminotransferase),
          total_protiens: parseFloat(form.total_protiens),
          albumin: parseFloat(form.albumin),
          albumin_and_globulin_ratio: parseFloat(form.albumin_and_globulin_ratio),
        }),
      });
      if (!res.ok) throw new Error('Prediction failed');
      const data = await res.json();
      setResults(data);

      // Record prediction to serial timeline
      setTimeline(prev => [
        ...prev,
        {
          day: prev.length + 1,
          probability: data.probability,
          tier: data.tier,
          color: data.color,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          labs: { ...form },
          summary: data.summary,
          actions: data.actions,
          abnormal_markers: data.abnormal_markers,
          shap_contributions: data.shap_contributions,
          model_metrics: data.model_metrics
        }
      ]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [form]);

  useEffect(() => { analyze(); }, []);

  // Educational Visualizer for SHAP
  const renderDiagnosticBalance = () => {
    if (!results || !results.shap_contributions) return null;
    
    return (
      <div className="diagnostic-balance">
        <h3 className="section-title"><HelpCircle size={16}/> Pathological Weighting</h3>
        <p className="edu-text">This visualizes how specific biomarkers tilt the diagnosis toward or away from disease.</p>
        
        <div className="balance-scale">
          <div className="scale-header">
            <span className="safe-text">Healthy Indicators</span>
            <span className="risk-text">Pathology Indicators</span>
          </div>
          
          <div className="scale-tracks">
            {results.shap_contributions.map((s) => {
              const isRisk = s.shap_value > 0;
              const width = Math.min(Math.abs(s.shap_value) * 150, 100);
              
              return (
                <div key={s.feature} className="scale-row">
                  <div className="scale-label">
                    <span className="marker-name">{s.label}</span>
                    <span className="marker-val">{s.value} <small>{s.unit}</small></span>
                  </div>
                  
                  <div className="scale-bar-area">
                    {/* Left side (Healthy) */}
                    <div className="scale-side left">
                      {!isRisk && <div className="bar safe-bar" style={{ width: `${width}%` }} />}
                    </div>
                    
                    <div className="scale-center-line" />
                    
                    {/* Right side (Risk) */}
                    <div className="scale-side right">
                      {isRisk && <div className="bar risk-bar" style={{ width: `${width}%` }} />}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="edu-layout">
      
      {/* LEFT COLUMN: The Patient Chart */}
      <section className="patient-chart custom-scrollbar">
        <header className="chart-header">
          <div className="academic-brand">
            <Stethoscope size={28} className="brand-icon" />
            <div>
              <h1>HepSense Clinical</h1>
              <p>ILPD Diagnostic Educator</p>
            </div>
          </div>
          <div className={`status-pill ${apiStatus}`}>
            {apiStatus === 'ok' ? 'System Ready' : apiStatus === 'error' ? 'Engine Offline' : 'Initializing...'}
          </div>
        </header>

        <div className="chart-content">
          <div className="form-section">
            <h2 className="form-heading"><FileText size={16}/> Reference Cases</h2>
            <div className="preset-row">
              {PRESETS.map(p => (
                <button 
                  key={p.id} 
                  className={`preset-pill ${activePreset === p.id ? 'active' : ''}`}
                  onClick={() => applyPreset(p)}
                >
                  {p.name}
                </button>
              ))}
            </div>
          </div>

          <div className="form-section">
            <h2 className="form-heading"><UserCircle size={16}/> Patient Vitals</h2>
            <div className="input-grid-2">
              <div className="edu-input">
                <label>Age</label>
                <input type="number" value={form.age} onChange={e => setField('age', e.target.value)} />
              </div>
              <div className="edu-input">
                <label>Biological Sex</label>
                <select value={form.gender} onChange={e => setField('gender', e.target.value)}>
                  <option>Male</option><option>Female</option>
                </select>
              </div>
            </div>
          </div>

          <div className="form-section">
            <h2 className="form-heading"><FlaskConical size={16}/> Comprehensive Hepatic Panel</h2>
            <p className="edu-text">Enter biomarker assays to evaluate hepatic synthetic and excretory function.</p>
            <div className="input-grid-2">
              {FIELDS.map(({ key, label, unit, step }) => (
                <div key={key} className="edu-input">
                  <label>{label} <span>{unit}</span></label>
                  <input
                    type="number" step={step} value={form[key]}
                    onChange={e => setField(key, e.target.value)}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="chart-footer">
          <button className="edu-btn-primary" onClick={analyze} disabled={loading}>
            {loading ? 'Synthesizing Data...' : 'Synthesize Diagnosis'}
          </button>
          {error && <div className="edu-error"><AlertCircle size={14}/> {error}</div>}
        </div>
      </section>


      {/* RIGHT COLUMN: The Diagnostic Whiteboard */}
      <section className="diagnostic-whiteboard custom-scrollbar">
        {!results ? (
          <div className="whiteboard-empty">
            <ShieldCheck size={48} className="empty-icon" />
            <h2>Awaiting Patient Data</h2>
            <p>Input clinical parameters on the left to generate an explainable diagnostic assessment.</p>
          </div>
        ) : (
          <div className="whiteboard-content fade-in">
            
            <div className="synthesis-header">
              <h2 className="serif-title">Diagnostic Synthesis</h2>
              <p className="edu-text">
                Model Confidence: <strong>ROC AUC {modelInfo?.roc_auc ?? '0.804'}</strong> ({modelInfo?.metadata?.dataset_name || 'ILPD Cohort'})
              </p>
            </div>

            <div className="verdict-card">
              <div className="verdict-score">
                <div className="score-circle" data-tier={results.color}>
                  {results.probability}%
                </div>
                <div className="score-text">
                  <span className="tier-label" data-tier={results.color}>{results.tier}</span>
                  <span className="prob-label">Calculated Probability</span>
                </div>
              </div>
              <div className="verdict-summary">
                {results.summary}
              </div>
            </div>

            {/* Serial Clinical Surveillance Timeline */}
            <div className="abnormal-flags">
              <div className="section-header-row" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <h3 className="section-title" style={{ margin: 0 }}><Activity size={16}/> Clinical Surveillance Timeline</h3>
                {timeline.length > 0 && (
                  <button 
                    type="button"
                    className="timeline-reset-btn"
                    onClick={() => setTimeline([])}
                  >
                    Clear Timeline
                  </button>
                )}
              </div>
              <p className="edu-text" style={{ marginBottom: '20px' }}>
                Surveillance tracking for longitudinal stays (e.g. 14 days). Click any day node to restore that historical run and its parameters.
              </p>

              <div className="timeline-container">
                {timeline.length === 0 ? (
                  <div className="timeline-empty-msg">
                    No timeline tracking data. Click 'Synthesize Diagnosis' to record serial entries.
                  </div>
                ) : (
                  <div className="timeline-track-wrapper">
                    <div className="timeline-line" />
                    <div className="timeline-nodes">
                      {timeline.map((point) => (
                        <button
                          key={point.day}
                          type="button"
                          className="timeline-node-item"
                          onClick={() => {
                            setForm(point.labs);
                            setResults({
                              probability: point.probability,
                              tier: point.tier,
                              color: point.color,
                              summary: point.summary,
                              actions: point.actions,
                              abnormal_markers: point.abnormal_markers,
                              shap_contributions: point.shap_contributions,
                              model_metrics: point.model_metrics
                            });
                          }}
                        >
                          <div className={`timeline-node-circle ${point.color}`}>
                            {point.probability}%
                          </div>
                          <span className="timeline-node-day">Day {point.day}</span>
                          <span className="timeline-node-time">{point.timestamp}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {results.abnormal_markers?.length > 0 && (
              <div className="abnormal-flags">
                <h3 className="section-title"><AlertCircle size={16}/> Out-of-Range Assays</h3>
                <div className="flag-grid">
                  {results.abnormal_markers.map(m => (
                    <div key={m.feature} className="flag-item">
                      <div className="flag-name">{m.label}</div>
                      <div className="flag-val">
                        {m.value} {m.unit}
                        <span className={`flag-dir ${m.direction}`}>{m.direction === 'high' ? '↑' : '↓'}</span>
                      </div>
                      <div className="flag-ref">Normal: {m.normal}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {renderDiagnosticBalance()}

            <div className="clinical-plan">
              <h3 className="section-title"><Activity size={16}/> Recommended Pathway</h3>
              <ul className="plan-list">
                {results.actions.map((action, i) => (
                  <li key={i}>
                    <div className="plan-step">{i + 1}</div>
                    <span>{action}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Collapsible Advanced Diagnostics Drawer */}
            <div className="advanced-toggle-sec">
              <button 
                type="button"
                className="advanced-toggle-btn"
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                <span>Advanced Validation & Calibration Details</span>
                {showAdvanced ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
              </button>

              {showAdvanced && (
                <div className="advanced-panel-content fade-in">
                  <div className="metrics-row">
                    <div className="metric-pill">
                      <span className="mv">{results.model_metrics.roc_auc}</span>
                      <span className="ml">ROC AUC</span>
                    </div>
                    <div className="metric-pill">
                      <span className="mv">{results.model_metrics.pr_auc}</span>
                      <span className="ml">PR AUC</span>
                    </div>
                    <div className="metric-pill">
                      <span className="mv">{(results.model_metrics.threshold * 100).toFixed(1)}%</span>
                      <span className="ml">Threshold</span>
                    </div>
                    <div className="metric-pill">
                      <span className="mv">{modelInfo?.metadata?.n_samples || 583}</span>
                      <span className="ml">Cohort Size</span>
                    </div>
                  </div>

                  {modelInfo?.metadata && (
                    <ul className="metadata-list" style={{ marginTop: '16px' }}>
                      <li className="metadata-item">
                        <ChevronRight size={12} className="bullet-icon" />
                        <span>Classifier: <strong>{modelInfo.metadata.algorithm}</strong></span>
                      </li>
                      <li className="metadata-item">
                        <ChevronRight size={12} className="bullet-icon" />
                        <span>Probability Calibration: <strong>{modelInfo.metadata.calibration}</strong></span>
                      </li>
                      <li className="metadata-item">
                        <ChevronRight size={12} className="bullet-icon" />
                        <span>Validation Protocol: <strong>{modelInfo.metadata.validation}</strong></span>
                      </li>
                      <li className="metadata-item">
                        <ChevronRight size={12} className="bullet-icon" />
                        <span>Threshold Criterion: <strong>{modelInfo.metadata.threshold_criterion}</strong></span>
                      </li>
                    </ul>
                  )}
                </div>
              )}
            </div>

          </div>
        )}
      </section>

    </div>
  );
}