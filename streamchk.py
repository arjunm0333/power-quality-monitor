import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.models import load_model  # type: ignore
from scipy.fft import fft

st.set_page_config(page_title="PQ Monitor — Fault Detection", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif !important; background-color: #0a0c10 !important; color: #e2e8f0 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1400px; }
section[data-testid="stSidebar"] { background: #0f1117 !important; border-right: 1px solid #21273a !important; }
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }
.stSlider [data-testid="stThumbValue"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.75rem !important; }
[data-testid="stMetric"] { background: #161b27 !important; border: 1px solid #21273a !important; border-radius: 6px !important; padding: 14px 16px !important; }
[data-testid="stMetricLabel"] { font-size: 0.68rem !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; color: #8892a4 !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 1.2rem !important; font-weight: 600 !important; color: #e2e8f0 !important; }
.stAlert { border-radius: 4px !important; border-width: 1px !important; font-size: 0.88rem !important; }
.streamlit-expanderHeader { background: #161b27 !important; border: 1px solid #21273a !important; border-radius: 6px !important; font-size: 0.78rem !important; font-weight: 600 !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; color: #8892a4 !important; }
.streamlit-expanderContent { border: 1px solid #21273a !important; border-top: none !important; background: #0f1117 !important; }
.stTable table { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important; border-collapse: collapse !important; }
.stTable th { background: #161b27 !important; color: #8892a4 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; font-size: 0.68rem !important; padding: 8px 12px !important; border-bottom: 1px solid #21273a !important; }
.stTable td { padding: 8px 12px !important; border-bottom: 1px solid #21273a !important; }
.stProgress > div > div { border-radius: 2px !important; }
.stProgress > div { border-radius: 2px !important; background: #21273a !important; height: 4px !important; }
div[data-testid="stInfo"]    { background: rgba(59,130,246,0.07) !important; border-color: rgba(59,130,246,0.25) !important; }
div[data-testid="stSuccess"] { background: rgba(16,185,129,0.07) !important; border-color: rgba(16,185,129,0.25) !important; }
div[data-testid="stWarning"] { background: rgba(245,158,11,0.07) !important; border-color: rgba(245,158,11,0.25) !important; }
div[data-testid="stError"]   { background: rgba(239,68,68,0.07)  !important; border-color: rgba(239,68,68,0.25)  !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ai_model():
    try: return load_model("pq_model.h5")
    except: return None

model = load_ai_model()

def section_heading(title):
    st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#6b84a0;margin-bottom:1rem;border-bottom:1px solid #21273a;padding-bottom:0.4rem;">{title}</div>', unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">
  <div style="width:3px;height:36px;background:linear-gradient(180deg,#3b82f6,#06b6d4);border-radius:2px;flex-shrink:0;"></div>
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:1.05rem;font-weight:600;letter-spacing:0.07em;text-transform:uppercase;color:#e2e8f0;">Power Quality Monitoring System</div>
    <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.73rem;color:#4a5568;letter-spacing:0.03em;margin-top:3px;">Real-time fault detection &nbsp;·&nbsp; CNN + LSTM classifier &nbsp;·&nbsp; IEEE 1159 thresholds</div>
  </div>
</div>
<div style="height:1px;background:linear-gradient(90deg,#21273a 60%,transparent);margin:14px 0 22px;"></div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#4a5568;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid #21273a;">Signal Parameters</div>', unsafe_allow_html=True)

amplitude = st.sidebar.slider("Voltage Amplitude (pu)", 0.3, 1.5, 1.0, 0.01)
frequency  = st.sidebar.slider("Frequency (Hz)", 45, 55, 50)
harmonic   = st.sidebar.slider("Harmonic Level (pu)", 0.0, 0.6, 0.0, 0.01)

model_status = "LOADED" if model is not None else "OFFLINE"
model_color  = "#10b981" if model is not None else "#ef4444"
st.sidebar.markdown(f"""
<div style="height:1px;background:#21273a;margin:18px 0;"></div>
<div style="background:#0a0c10;border:1px solid #21273a;border-radius:8px;overflow:hidden;margin-bottom:16px;">
  <div style="padding:8px 14px;border-bottom:1px solid #21273a;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#4a5568;">IEEE 1159 Thresholds</span>
  </div>
  <div style="padding:4px 0;">
    <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 14px;border-bottom:1px solid #0f1117;">
      <span style="font-family:'IBM Plex Sans',sans-serif;font-size:0.75rem;color:#64748b;">Normal</span>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:0.73rem;font-weight:500;color:#e2e8f0;background:#10b98114;border:1px solid #10b98130;padding:2px 8px;border-radius:3px;">0.9 – 1.1 pu</span>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 14px;border-bottom:1px solid #0f1117;">
      <span style="font-family:'IBM Plex Sans',sans-serif;font-size:0.75rem;color:#64748b;">Voltage Sag</span>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:0.73rem;font-weight:500;color:#f59e0b;background:#f59e0b14;border:1px solid #f59e0b30;padding:2px 8px;border-radius:3px;">&lt; 0.9 pu</span>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 14px;border-bottom:1px solid #0f1117;">
      <span style="font-family:'IBM Plex Sans',sans-serif;font-size:0.75rem;color:#64748b;">Voltage Swell</span>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:0.73rem;font-weight:500;color:#f59e0b;background:#f59e0b14;border:1px solid #f59e0b30;padding:2px 8px;border-radius:3px;">&gt; 1.1 pu</span>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 14px;">
      <span style="font-family:'IBM Plex Sans',sans-serif;font-size:0.75rem;color:#64748b;">Harmonic THD</span>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:0.73rem;font-weight:500;color:#ef4444;background:#ef444414;border:1px solid #ef444430;padding:2px 8px;border-radius:3px;">≥ 5%</span>
    </div>
  </div>
</div>
<div style="display:flex;align-items:center;justify-content:space-between;padding:10px 14px;background:#0a0c10;border:1px solid #21273a;border-radius:8px;">
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:3px;">CNN + LSTM Model</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;font-weight:600;color:{model_color};">{model_status}</div>
  </div>
  <div style="width:8px;height:8px;border-radius:50%;background:{model_color};box-shadow:0 0 10px {model_color}88;"></div>
</div>
""", unsafe_allow_html=True)

# Signal generation
timesteps = 1000
t      = np.linspace(0, 1, timesteps)
signal = amplitude * np.sin(2 * np.pi * frequency * t)
if harmonic > 0:
    signal += harmonic * np.sin(2 * np.pi * 3 * frequency * t)

rms    = np.sqrt(np.mean(signal**2))
peak   = np.max(np.abs(signal))
mean_v = np.mean(signal)
std_v  = np.std(signal)

fft_vals = np.abs(fft(signal))[:len(signal)//2]
fundamental_idx = int(frequency * timesteps / (timesteps / t[-1]))
fundamental = fft_vals[fundamental_idx] if 0 < fundamental_idx < len(fft_vals) else fft_vals[1]
harmonics_sq = sum(fft_vals[i]**2 for i in range(2, min(20, len(fft_vals))) if i != fundamental_idx)
thd = (np.sqrt(harmonics_sq) / fundamental * 100) if fundamental > 1e-10 else 0

# Waveform
section_heading("Voltage Waveform")

matplotlib.rcParams.update({
    'font.family': 'monospace', 'axes.facecolor': '#080b12', 'figure.facecolor': '#080b12',
    'axes.edgecolor': '#1a2035', 'axes.labelcolor': '#c8d6e8',
    'xtick.color': '#c8d6e8', 'ytick.color': '#c8d6e8',
    'grid.color': '#111827', 'text.color': '#94a3b8',
})

if thd >= 5.0:       glow_color = '#ff2d55'
elif amplitude < 0.9 or amplitude > 1.1: glow_color = '#ffcc00'
else:                glow_color = '#00f5c4'

fig, ax = plt.subplots(figsize=(14, 4.2), facecolor='#04060d')
ax.set_facecolor('#04060d')

ax.axhspan( 1.1,  2.0, alpha=0.04, color='#ffcc00', zorder=0)
ax.axhspan(-2.0, -1.1, alpha=0.04, color='#ffcc00', zorder=0)
ax.axhspan(-0.9,  0.9, alpha=0.015, color='#00f5c4', zorder=0)
ax.axhline(y= 1.1, color='#ffcc00', linestyle='--', linewidth=0.7, alpha=0.4, zorder=1)
ax.axhline(y=-1.1, color='#ffcc00', linestyle='--', linewidth=0.7, alpha=0.4, zorder=1)
ax.axhline(y= 0.9, color='#00f5c4', linestyle=':',  linewidth=0.6, alpha=0.3, zorder=1)
ax.axhline(y=-0.9, color='#00f5c4', linestyle=':',  linewidth=0.6, alpha=0.3, zorder=1)
ax.axhline(y= 0,   color='#0d1520', linestyle='-',  linewidth=0.8, alpha=1.0, zorder=1)

ax.fill_between(t, signal, 0, where=(signal >= 0), alpha=0.07, color=glow_color, zorder=2)
ax.fill_between(t, signal, 0, where=(signal <  0), alpha=0.05, color=glow_color, zorder=2)
ax.plot(t, signal, linewidth=12.0, color=glow_color, alpha=0.03, zorder=3, solid_capstyle='round')
ax.plot(t, signal, linewidth=6.0,  color=glow_color, alpha=0.07, zorder=3, solid_capstyle='round')
ax.plot(t, signal, linewidth=3.0,  color=glow_color, alpha=0.18, zorder=4, solid_capstyle='round')
ax.plot(t, signal, linewidth=1.6,  color=glow_color, alpha=0.70, zorder=5)
ax.plot(t, signal, linewidth=0.7,  color='#ffffff',  alpha=0.85, zorder=6)

ax.text(0.002,  1.13, '+1.1 pu  swell',  fontsize=7, color='#ffcc00', alpha=0.55, va='bottom', fontfamily='monospace')
ax.text(0.002, -1.14, '−1.1 pu  swell',  fontsize=7, color='#ffcc00', alpha=0.55, va='top',    fontfamily='monospace')
ax.text(0.002,  0.92, '+0.9 pu  normal', fontsize=7, color='#00f5c4', alpha=0.4,  va='bottom', fontfamily='monospace')

ax.set_xlabel("Time (s)", fontsize=9, labelpad=10, color='#c8d6e8', fontfamily='monospace')
ax.set_ylabel("Voltage (pu)", fontsize=9, labelpad=10, color='#c8d6e8', fontfamily='monospace')
ax.set_xlim([0, 1]); ax.set_ylim([-2, 2])
ax.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
ax.yaxis.grid(True, alpha=0.12, linewidth=0.5, color='#0d1a2e', zorder=0)
ax.xaxis.grid(True, alpha=0.07, linewidth=0.4, color='#0d1a2e', zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#1e2d42'); ax.spines['bottom'].set_color('#1e2d42')
ax.tick_params(labelsize=8.5, colors='#c8d6e8', length=4, which='both')
ax.text(0.99, 0.97, f"A={amplitude:.2f} pu   RMS={rms:.3f}   THD={thd:.1f}%   f={frequency} Hz",
        transform=ax.transAxes, fontsize=7.5, color='#6b84a0', ha='right', va='top', fontfamily='monospace')

plt.tight_layout(pad=0.6)
st.markdown('<div style="background:#080b12;border:1px solid #1a2035;border-radius:8px 8px 0 0;overflow:hidden;padding:0;">', unsafe_allow_html=True)
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)
plt.close()

st.markdown(f"""
<div style="display:flex;align-items:stretch;background:#080b12;border:1px solid #1a2035;border-top:none;border-radius:0 0 8px 8px;margin-bottom:4px;overflow:hidden;">
  <div style="display:flex;align-items:center;gap:10px;padding:9px 18px;border-right:1px solid #0d1420;flex:1;">
    <span style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;font-weight:700;color:#6b84a0;text-transform:uppercase;letter-spacing:0.08em;flex-shrink:0;">X-Axis</span>
    <span style="width:1px;height:14px;background:#1a2035;flex-shrink:0;"></span>
    <span style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#c8d6e8;font-weight:600;">Time (s)</span>
    <span style="font-family:IBM Plex Sans,sans-serif;font-size:0.7rem;color:#4a6080;">— Duration of the observed signal window. Range: 0 to 1 second (one full cycle period at {frequency} Hz).</span>
  </div>
  <div style="display:flex;align-items:center;gap:10px;padding:9px 18px;flex:1;">
    <span style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;font-weight:700;color:#6b84a0;text-transform:uppercase;letter-spacing:0.08em;flex-shrink:0;">Y-Axis</span>
    <span style="width:1px;height:14px;background:#1a2035;flex-shrink:0;"></span>
    <span style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#c8d6e8;font-weight:600;">Voltage (pu)</span>
    <span style="font-family:IBM Plex Sans,sans-serif;font-size:0.7rem;color:#4a6080;">— Instantaneous voltage in per-unit. 1.0 pu = nominal. Normal band: ±0.9–1.1 pu. Current peak: {peak:.3f} pu.</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Signal parameters
section_heading("Signal Parameters")
c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
c1.metric("Amplitude",   f"{amplitude:.3f} pu")
c2.metric("RMS",         f"{rms:.3f} pu")
c3.metric("Peak",        f"{peak:.3f} pu")
c4.metric("THD",         f"{thd:.2f}%")
c5.metric("Frequency",   f"{frequency} Hz")
c6.metric("Mean",        f"{mean_v:.4f}")
c7.metric("Std Dev",     f"{std_v:.4f}")
c8.metric("Harmonic In", f"{harmonic:.2f} pu")

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# Thresholds
SAG_LIMIT, SWELL_LIMIT, THD_LIMIT = 0.9, 1.1, 5.0

if thd >= THD_LIMIT:
    rule_result, trigger_reason = "Harmonic Distortion", f"THD = {thd:.2f}% (limit {THD_LIMIT}%)"
elif amplitude < SAG_LIMIT:
    rule_result, trigger_reason = "Voltage Sag",        f"Amplitude = {amplitude:.3f} pu (limit {SAG_LIMIT} pu)"
elif amplitude > SWELL_LIMIT:
    rule_result, trigger_reason = "Voltage Swell",      f"Amplitude = {amplitude:.3f} pu (limit {SWELL_LIMIT} pu)"
else:
    rule_result, trigger_reason = "Normal",             "All parameters within normal operating range"

# Fault detection cards
section_heading("Fault Detection")

PALETTE = {
    "Normal":              ("#10b981", "#0a1f15"),
    "Voltage Sag":         ("#f59e0b", "#1f160a"),
    "Voltage Swell":       ("#f59e0b", "#1f160a"),
    "Harmonic Distortion": ("#ef4444", "#1f0a0a"),
}

def result_card(method, result, detail="", sub=""):
    accent, bg = PALETTE.get(result, ("#8892a4", "#0f1117"))
    sub_html = f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;color:#4a5568;margin-top:3px;">{sub}</div>' if sub else ""
    return (f'<div style="background:{bg};border:1px solid {accent}33;border-left:3px solid {accent};border-radius:6px;padding:16px 18px;height:100%;box-sizing:border-box;">'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:{accent};opacity:0.65;margin-bottom:8px;">{method}</div>'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.05rem;font-weight:600;color:{accent};line-height:1.2;">{result}</div>'
            f'<div style="font-family:IBM Plex Sans,sans-serif;font-size:0.73rem;color:#8892a4;margin-top:7px;">{detail}</div>'
            f'{sub_html}</div>')

col_rule, col_ai, col_final = st.columns(3)
with col_rule:
    st.markdown(result_card("Rule-Based · IEEE 1159", rule_result, trigger_reason), unsafe_allow_html=True)

fault_dict = {0:"Normal", 1:"Voltage Sag", 2:"Voltage Swell", 3:"Harmonic Distortion"}

if model is not None:
    signal_input    = signal.reshape(1, timesteps, 1)
    prediction      = model.predict(signal_input, verbose=0)
    predicted_class = np.argmax(prediction)
    ai_result       = fault_dict[predicted_class]
    ai_conf         = float(np.max(prediction[0])) * 100
    second_idx      = np.argsort(prediction[0])[-2]

    with col_ai:
        st.markdown(result_card(
            "AI Classifier · CNN + LSTM", ai_result,
            f"Confidence: {ai_conf:.1f}%",
            f"2nd: {fault_dict[second_idx]} ({prediction[0][second_idx]*100:.1f}%)"
        ), unsafe_allow_html=True)

    if rule_result == ai_result:
        final_decision, decision_method, confidence_level = rule_result, "Consensus", ai_conf
    else:
        if ai_conf > 90:
            final_decision, decision_method, confidence_level = ai_result, "AI — High Confidence", ai_conf
        elif thd >= THD_LIMIT * 1.5:
            final_decision, decision_method, confidence_level = "Harmonic Distortion", "Critical Override", 95.0
        elif amplitude < SAG_LIMIT * 0.85:
            final_decision, decision_method, confidence_level = "Voltage Sag", "Critical Override", 95.0
        elif amplitude > SWELL_LIMIT * 1.15:
            final_decision, decision_method, confidence_level = "Voltage Swell", "Critical Override", 95.0
        elif ai_conf > 70:
            vpass = (
                (ai_result == "Harmonic Distortion" and (thd >= THD_LIMIT * 0.8 or harmonic > 0.15)) or
                (ai_result == "Voltage Sag"         and (amplitude < SAG_LIMIT * 1.05 or rms < 0.9)) or
                (ai_result == "Voltage Swell"       and (amplitude > SWELL_LIMIT * 0.95 or rms > 1.1)) or
                (ai_result == "Normal"              and SAG_LIMIT*0.95 <= amplitude <= SWELL_LIMIT*1.05 and thd < THD_LIMIT*0.8)
            )
            if vpass: final_decision, decision_method, confidence_level = ai_result, "AI — Verified", ai_conf
            else:     final_decision, decision_method, confidence_level = rule_result, "Rule-Based (AI Unverified)", 75.0
        else:
            final_decision, decision_method, confidence_level = rule_result, "Rule-Based (Low AI Conf.)", 70.0

    with col_final:
        st.markdown(result_card(f"Final Diagnosis · {decision_method}", final_decision,
                                f"System confidence: {confidence_level:.1f}%"), unsafe_allow_html=True)
else:
    ai_result, ai_conf, prediction, predicted_class = "N/A", 0.0, None, 0
    final_decision, decision_method, confidence_level = rule_result, "Rule-Based (No Model)", 70.0
    with col_ai:
        st.markdown('<div style="background:#0f1117;border:1px solid #21273a;border-radius:6px;padding:16px 18px;"><div style="font-family:IBM Plex Mono,monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#4a5568;margin-bottom:8px;">AI Classifier · CNN + LSTM</div><div style="font-family:IBM Plex Mono,monospace;font-size:0.9rem;color:#2d3650;">Model offline</div></div>', unsafe_allow_html=True)
    with col_final:
        st.markdown(result_card("Final Diagnosis · Rule-Based", final_decision, trigger_reason), unsafe_allow_html=True)

# Confidence bar
bar_color = "#10b981" if final_decision == "Normal" else ("#ef4444" if final_decision == "Harmonic Distortion" else "#f59e0b")
st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin:14px 0 10px;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.08em;width:130px;flex-shrink:0;">System Confidence</div>
  <div style="flex:1;height:3px;background:#1c2333;border-radius:2px;overflow:hidden;">
    <div style="width:{confidence_level:.1f}%;height:100%;background:{bar_color};border-radius:2px;"></div>
  </div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;font-weight:600;color:{bar_color};width:50px;text-align:right;">{confidence_level:.1f}%</div>
</div>
""", unsafe_allow_html=True)

advisory = {
    "Normal":              ("System operating within normal parameters. No action required.", "info"),
    "Voltage Sag":         (f"Voltage drop of {((1.0-amplitude)*100):.1f}%. Investigate heavy loads and supply integrity. Consider voltage regulation.", "warning"),
    "Voltage Swell":       (f"Voltage rise of {((amplitude-1.0)*100):.1f}%. Check load disconnection and capacitor banks. Protect sensitive equipment.", "warning"),
    "Harmonic Distortion": (f"THD at {thd:.2f}% — exceeds {THD_LIMIT}% limit. Identify harmonic sources and implement passive or active filtering.", "error"),
}
msg, level = advisory.get(final_decision, ("Unknown fault condition.", "warning"))
getattr(st, level)(f"**Advisory:** {msg}")

if model is not None and prediction is not None:
    with st.expander("AI Probability Distribution"):
        fig3, ax3 = plt.subplots(figsize=(10, 2.4))
        classes = list(fault_dict.values())
        probs   = prediction[0] * 100
        c_bars  = ['#3b82f6' if i == predicted_class else '#161b27' for i in range(4)]
        c_edge  = ['#3b82f6' if i == predicted_class else '#21273a' for i in range(4)]
        bars = ax3.barh(classes, probs, color=c_bars, edgecolor=c_edge, linewidth=0.8, height=0.5)
        for bar, val in zip(bars, probs):
            ax3.text(min(val+1.5, 96), bar.get_y()+bar.get_height()/2,
                     f'{val:.1f}%', va='center', ha='left', fontsize=8.5, color='#e2e8f0', fontfamily='monospace')
        ax3.set_xlim([0, 108]); ax3.set_xlabel("Confidence (%)", fontsize=8.5)
        ax3.grid(axis='x', alpha=0.3, linewidth=0.5)
        ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
        ax3.tick_params(labelsize=8.5)
        plt.tight_layout(pad=0.5); st.pyplot(fig3); plt.close()

with st.expander("Rule-Based Detection Logic"):
    col_t, col_m = st.columns(2)
    with col_t:
        st.markdown("**Threshold Table**")
        st.table({
            "Parameter":    ["Amplitude", "THD"],
            "Measured":     [f"{amplitude:.3f} pu", f"{thd:.2f}%"],
            "Normal Range": [f"{SAG_LIMIT}–{SWELL_LIMIT} pu", f"< {THD_LIMIT}%"],
            "Status":       ["Normal" if SAG_LIMIT <= amplitude <= SWELL_LIMIT else "Abnormal",
                             "Normal" if thd < THD_LIMIT else "Abnormal"],
        })
    with col_m:
        st.markdown("**Decision Tree**")
        st.code("if THD >= 5.0%        → Harmonic Distortion\nelif Amplitude < 0.9  → Voltage Sag\nelif Amplitude > 1.1  → Voltage Swell\nelse                  → Normal", language="python")
        st.caption(f"Trigger: {trigger_reason}")

# Error metrics
if model is not None and prediction is not None:
    section_heading("Model Error Metrics")

    y_true_oh   = np.zeros(4); y_true_oh[predicted_class] = 1.0
    y_pred_prob = prediction[0]
    prob_rmse   = float(np.sqrt(np.mean((y_true_oh - y_pred_prob) ** 2)))
    max_conf    = float(np.max(y_pred_prob))
    cer         = (1.0 - max_conf) * 100.0
    y_clip        = np.clip(y_pred_prob, 1e-10, 1 - 1e-10)
    cross_entropy = float(-np.sum(y_true_oh * np.log(y_clip)))
    entropy     = float(-np.sum(y_pred_prob * np.log(np.clip(y_pred_prob, 1e-10, 1))))
    sorted_p    = np.sort(y_pred_prob)[::-1]
    margin      = float((sorted_p[0] - sorted_p[1]) * 100)

    def badge(label, color):
        return f'<span style="background:{color}1a;color:{color};border:1px solid {color}40;padding:2px 9px;border-radius:3px;font-size:0.65rem;font-weight:600;letter-spacing:0.07em;font-family:IBM Plex Mono,monospace;">{label}</span>'
    def rmse_badge(v):
        if v < 0.05: return badge("EXCELLENT", "#10b981")
        if v < 0.15: return badge("GOOD",      "#3b82f6")
        if v < 0.30: return badge("FAIR",      "#f59e0b")
        return badge("POOR", "#ef4444")
    def cer_badge(v):
        if v < 5:  return badge("EXCELLENT", "#10b981")
        if v < 20: return badge("GOOD",      "#3b82f6")
        if v < 40: return badge("FAIR",      "#f59e0b")
        return badge("POOR", "#ef4444")
    def ce_badge(v):
        if v < 0.1: return badge("EXCELLENT", "#10b981")
        if v < 0.5: return badge("GOOD",      "#3b82f6")
        if v < 1.0: return badge("FAIR",      "#f59e0b")
        return badge("HIGH LOSS", "#ef4444")

    def metric_card(title, subtitle, value, unit, badge_html, formula_lhs, formula_rhs, note, accent):
        return (f'<div style="background:#0f1117;border:1px solid #21273a;border-top:2px solid {accent};border-radius:8px;padding:22px 20px 18px;height:100%;box-sizing:border-box;">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;gap:8px;">'
                f'<div><div style="font-family:IBM Plex Sans,sans-serif;font-size:0.84rem;font-weight:600;color:#e2e8f0;">{title}</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.6rem;color:#4a6080;margin-top:3px;text-transform:uppercase;letter-spacing:0.08em;">{subtitle}</div></div>'
                f'<div style="flex-shrink:0;margin-top:2px;">{badge_html}</div></div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:2.1rem;font-weight:700;color:{accent};letter-spacing:-1.5px;line-height:1;">'
                f'{value}<span style="font-size:0.82rem;font-weight:400;color:#4a6080;margin-left:4px;">{unit}</span></div>'
                f'<div style="display:flex;align-items:center;margin:14px 0 12px;padding:10px 14px;background:#0a0c10;border-radius:5px;border:1px solid #1a2035;">'
                f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.82rem;font-weight:600;color:{accent};margin-right:8px;flex-shrink:0;">{formula_lhs}</span>'
                f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.82rem;color:#4a6080;">=</span>'
                f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.82rem;color:#8da4c0;margin-left:8px;">{formula_rhs}</span></div>'
                f'<div style="font-family:IBM Plex Sans,sans-serif;font-size:0.74rem;color:#64788f;line-height:1.6;">{note}</div></div>')

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(metric_card("Probability RMSE", "Confidence error", f"{prob_rmse:.4f}", "",
            rmse_badge(prob_rmse), "RMSE", "√( Σ(yₜ − ŷ)² / n )",
            "Distance between the softmax vector and the one-hot ground truth. Decreases as prediction confidence increases.", "#3b82f6"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("Classification Error Rate", "Uncertainty proxy", f"{cer:.2f}", "%",
            cer_badge(cer), "CER", "( 1 − max_conf ) × 100",
            "Residual uncertainty after the top-class confidence is taken. Classes: Normal · Sag · Swell · Harmonic.", "#f59e0b"), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card("Cross-Entropy Loss", "Classification loss", f"{cross_entropy:.4f}", "",
            ce_badge(cross_entropy), "H(y,ŷ)", "−Σ yₜ · log( ŷ )",
            "Primary CNN+LSTM training objective. Penalises the log-probability assigned to the correct class.", "#10b981"), unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    param_evidence = [
        ("Amplitude",   f"{amplitude:.3f} pu", f"{SAG_LIMIT}–{SWELL_LIMIT} pu",
         "Normal" if SAG_LIMIT <= amplitude <= SWELL_LIMIT else ("Voltage Sag" if amplitude < SAG_LIMIT else "Voltage Swell")),
        ("THD",         f"{thd:.2f}%",          f"< {THD_LIMIT}%",
         "Normal" if thd < THD_LIMIT else "Harmonic Distortion"),
        ("RMS Voltage", f"{rms:.4f} pu",        "0.9–1.1 pu",
         "Normal" if 0.9 <= rms <= 1.1 else ("Voltage Sag" if rms < 0.9 else "Voltage Swell")),
        ("Harmonic In", f"{harmonic:.3f} pu",   "< 0.1 pu",
         "Normal" if harmonic < 0.1 else "Harmonic Distortion"),
        ("Frequency",   f"{frequency} Hz",       "49–51 Hz",
         "Normal" if 49 <= frequency <= 51 else "Deviation"),
    ]

    rule_support = sum(1 for _,_,_, v in param_evidence if v == rule_result or (rule_result == "Normal" and v == "Normal"))
    ai_support   = sum(1 for _,_,_, v in param_evidence if v == ai_result  or (ai_result  == "Normal" and v == "Normal"))

    COLORS_MAP = {"Normal":"#10b981","Voltage Sag":"#f59e0b","Voltage Swell":"#f59e0b","Harmonic Distortion":"#ef4444","Deviation":"#f59e0b"}

    def verdict_pill(verdict):
        c = COLORS_MAP.get(verdict, "#64748b")
        icon = "✓" if verdict == "Normal" else "!"
        return (f'<span style="background:{c}15;color:{c};border:1px solid {c}35;padding:2px 8px;border-radius:3px;'
                f'font-size:0.65rem;font-weight:600;font-family:IBM Plex Mono,monospace;white-space:nowrap;">{icon} {verdict}</span>')

    rows_html = ""
    for param, measured, normal_range, verdict in param_evidence:
        v_color = COLORS_MAP.get(verdict, "#64748b")
        row_bg  = f"{v_color}08" if verdict != "Normal" else "transparent"
        rows_html += (f'<tr style="border-bottom:1px solid #0d1420;background:{row_bg};">'
                      f'<td style="padding:9px 14px;font-family:IBM Plex Sans,sans-serif;font-size:0.76rem;color:#8da4c0;">{param}</td>'
                      f'<td style="padding:9px 14px;font-family:IBM Plex Mono,monospace;font-size:0.76rem;font-weight:600;color:#e2e8f0;">{measured}</td>'
                      f'<td style="padding:9px 14px;font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#6b84a0;">{normal_range}</td>'
                      f'<td style="padding:9px 14px;">{verdict_pill(verdict)}</td></tr>')

    rule_pct = int((rule_support / 5) * 100)
    ai_pct   = int((ai_support   / 5) * 100)
    rule_c   = COLORS_MAP.get(rule_result,    "#64748b")
    ai_c     = COLORS_MAP.get(ai_result,      "#64748b")
    final_c  = COLORS_MAP.get(final_decision, "#64748b")
    ai_display = ai_result if ai_result != "N/A" else "—"

    reason_map = {
        "Consensus":                  f"Both methods independently agree on <b>{final_decision}</b>. High reliability.",
        "AI — High Confidence":       "AI confidence exceeded 90% — deep pattern recognition overrides rule-based result.",
        "AI — Verified":              f"AI result cross-validated by {ai_support}/5 signal parameters. Parameter evidence supports AI.",
        "Critical Override":          "One or more parameters breached critical safety thresholds. Rule-based takes precedence.",
        "Rule-Based (AI Unverified)": f"AI result unverified by parameters ({ai_support}/5 match). Rule-based is more reliable here.",
        "Rule-Based (Low AI Conf.)":  "AI confidence below 70% — insufficient certainty. Rule-based threshold logic applied.",
        "Rule-Based (No Model)":      "No AI model loaded. Decision based entirely on IEEE 1159 threshold rules.",
    }
    reason_text = reason_map.get(decision_method, f"Decision made via {decision_method}.")

    panel_html = (
        f'<div style="background:#0a0d14;border:1px solid #1a2035;border-radius:10px;overflow:hidden;margin-top:4px;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;padding:13px 18px;border-bottom:1px solid #0d1420;background:#080b12;">'
        f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#6b84a0;">Decision Evidence &amp; Reasoning</span>'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#6b84a0;">Final:</span>'
        f'<span style="background:{final_c}18;color:{final_c};border:1px solid {final_c}40;padding:3px 10px;border-radius:4px;font-family:IBM Plex Mono,monospace;font-size:0.72rem;font-weight:700;">{final_decision}</span>'
        f'</div></div>'
        f'<div style="display:grid;grid-template-columns:1fr 270px;">'
        f'<div style="border-right:1px solid #0d1420;">'
        f'<div style="padding:10px 14px 6px;font-family:IBM Plex Mono,monospace;font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:#4a6080;">Signal Parameter Analysis</div>'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr style="border-bottom:1px solid #0d1420;">'
        f'<th style="padding:6px 14px;font-family:IBM Plex Mono,monospace;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;color:#4a6080;text-align:left;font-weight:600;">Parameter</th>'
        f'<th style="padding:6px 14px;font-family:IBM Plex Mono,monospace;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;color:#4a6080;text-align:left;font-weight:600;">Measured</th>'
        f'<th style="padding:6px 14px;font-family:IBM Plex Mono,monospace;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;color:#4a6080;text-align:left;font-weight:600;">Normal Range</th>'
        f'<th style="padding:6px 14px;font-family:IBM Plex Mono,monospace;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;color:#4a6080;text-align:left;font-weight:600;">Diagnosis</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        f'<div style="padding:16px;">'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:#4a6080;margin-bottom:10px;">Method Comparison</div>'
        f'<div style="margin-bottom:12px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
        f'<span style="font-family:IBM Plex Sans,sans-serif;font-size:0.75rem;color:#8da4c0;">Rule-Based</span>'
        f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.73rem;color:{rule_c};font-weight:600;">{rule_result}</span></div>'
        f'<div style="height:4px;background:#0d1520;border-radius:2px;overflow:hidden;"><div style="width:{rule_pct}%;height:100%;background:{rule_c};border-radius:2px;"></div></div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#4a6080;margin-top:3px;">{rule_support}/5 parameters</div></div>'
        f'<div style="margin-bottom:16px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
        f'<span style="font-family:IBM Plex Sans,sans-serif;font-size:0.75rem;color:#8da4c0;">AI Classifier</span>'
        f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.73rem;color:{ai_c};font-weight:600;">{ai_display}</span></div>'
        f'<div style="height:4px;background:#0d1520;border-radius:2px;overflow:hidden;"><div style="width:{ai_pct}%;height:100%;background:{ai_c};border-radius:2px;"></div></div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#4a6080;margin-top:3px;">{ai_support}/5 params · conf. {ai_conf:.1f}%</div></div>'
        f'<div style="height:1px;background:#0f1520;margin:0 0 12px;"></div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:#4a6080;margin-bottom:8px;">Resolution</div>'
        f'<div style="background:{final_c}10;border:1px solid {final_c}30;border-radius:5px;padding:8px 10px;margin-bottom:10px;">'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.66rem;color:#8da4c0;margin-bottom:3px;">{decision_method}</div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.82rem;font-weight:700;color:{final_c};">{final_decision}</div></div>'
        f'<div style="font-family:IBM Plex Sans,sans-serif;font-size:0.73rem;color:#7a90a8;line-height:1.6;">{reason_text}</div>'
        f'</div></div></div>'
    )
    st.markdown(panel_html, unsafe_allow_html=True)

with st.expander("Debug"):
    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("**Input**")
        st.code(f"Amplitude : {amplitude}\nFrequency : {frequency} Hz\nHarmonic  : {harmonic}", language="yaml")
    with dc2:
        st.markdown("**Computed**")
        st.code(f"RMS  : {rms:.6f}\nTHD  : {thd:.6f}%\nPeak : {peak:.6f}", language="yaml")

st.markdown("""
<div style="height:1px;background:linear-gradient(90deg,transparent,#21273a 30%,#21273a 70%,transparent);margin:32px 0 16px;"></div>
<div style="text-align:center;font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#2d3650;letter-spacing:0.12em;">
  POWER QUALITY MONITORING SYSTEM &nbsp;·&nbsp; CNN + LSTM FAULT CLASSIFIER &nbsp;·&nbsp; IEEE 1159
</div>
""", unsafe_allow_html=True)