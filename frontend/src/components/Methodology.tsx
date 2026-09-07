import { ALL_LF_NAMES, CLASS_DISPLAY_NAMES, ClassKey } from '../types/hotspot';
import ArchitectureDiagram from './sections/ArchitectureDiagram';

const LF_DESCRIPTIONS: Record<string, { votes: string; evidence: string; threshold: string }> = {
  lf_industry_high_frp: { votes: 'industrial_fire', evidence: 'Industry proximity + high FRP', threshold: 'industry ≤ 20 km, FRP ≥ 5 MW' },
  lf_factory_proximity_thermal: { votes: 'industrial_fire', evidence: 'Factory proximity + elevated brightness', threshold: 'industry ≤ 20 km, brightness ≥ 310 K, confidence ≠ low' },
  lf_industrial_zone_cluster: { votes: 'industrial_fire', evidence: 'Dense industrial cluster + heat', threshold: 'industry ≤ 20 km, ≥ 2 sites within 5 km, FRP ≥ 5 MW' },
  lf_oil_gas_flare: { votes: 'gas_flare', evidence: 'Oil/gas site + night, static type, or steady flare heat', threshold: 'oil/gas ≤ 2 km + (night OR FIRMS type 3 OR FRP ≥ 5 MW)' },
  lf_refinery_flare: { votes: 'gas_flare', evidence: 'Refinery + night, static type, or persistence', threshold: 'refinery ≤ 2 km + (night OR type 3 OR persistence ≥ 0.3)' },
  lf_mining_thermal_activity: { votes: 'mining_activity', evidence: 'Mine + supporting thermal signal', threshold: 'mining ≤ 20 km + (FRP ≥ 5 MW OR brightness ≥ 310 K), confidence ≠ low' },
  lf_mining_high_confidence: { votes: 'mining_activity', evidence: 'Mining region + high confidence, away from factories', threshold: 'mining ≤ 3 km, high confidence, FRP ≥ 5 MW, industry ≥ 1.5 km away' },
  lf_agriculture_vegetation_fire: { votes: 'agricultural_burn', evidence: 'Cropland + vegetation-fire signal', threshold: 'agriculture ≤ 30 km, FRP ≤ 45 MW' },
  lf_agriculture_burn_context: { votes: 'agricultural_burn', evidence: 'Farmland + moderate stubble thermal band', threshold: 'agriculture ≤ 30 km, FRP 5–15 MW, brightness 310–345 K' },
  lf_forest_vegetation_fire: { votes: 'forest_natural_fire', evidence: 'Forest + isolated vegetation fire', threshold: 'forest ≤ 30 km, FRP ≥ 2 MW' },
  lf_strong_forest_fire: { votes: 'forest_natural_fire', evidence: 'Forest + high-intensity wildfire signal', threshold: 'forest ≤ 30 km, FRP ≥ 15 MW' },
  lf_static_industrial_heat: { votes: 'industrial_process_heat', evidence: 'Known static land source near industry', threshold: 'FIRMS type 3, industry ≤ 40 km, FRP ≤ 25 MW' },
  lf_nighttime_process_heat: { votes: 'industrial_process_heat', evidence: 'Nighttime steady low industrial heat', threshold: 'night, FRP ≤ 10 MW, brightness ≥ 310 K, industry ≤ 20 km' },
  lf_power_plant_process_heat: { votes: 'industrial_process_heat', evidence: 'Power plant nearest + supporting heat', threshold: 'power plant ≤ 15 km and nearest, FRP ≥ 2 MW, brightness ≥ 310 K' },
};

const FUSION_CASES = [
  { name: 'Case A — hybrid_agreement (CONFIRMED)', desc: 'Independent rule evidence and ML agree. High confidence, auto-classified, no review needed.' },
  { name: 'Case B — ml_assisted (PROBABLE)', desc: 'Rules abstain but a single strong ML signal stands uncontested. Moderate confidence, auto-classified with ML justification.' },
  { name: 'Case C — rule_dominant (CONFIRMED)', desc: 'Strong multi-rule consensus (≥ 2 votes) with weak ML. Rules override uncertain ML.' },
  { name: 'Case D — conflict (UNCERTAIN)', desc: 'Rules and ML confidently disagree (strong/close-tie or ML-dominant sub-cases). Always flagged for human review with both perspectives shown.' },
  { name: 'Case E — uncertain', desc: 'All rules abstain and ML is weak. Output unclassified, always flagged for human review.' },
];

const ORDER: ClassKey[] = [
  'industrial_fire',
  'gas_flare',
  'mining_activity',
  'agricultural_burn',
  'forest_natural_fire',
  'industrial_process_heat',
  'unclassified',
];

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-2xl shadow p-5">
      <h3 className="font-bold text-slate-900 dark:text-slate-100 text-base mb-2">{title}</h3>
      <div className="text-sm text-slate-700 dark:text-slate-300 space-y-2 leading-relaxed">{children}</div>
    </section>
  );
}

export default function Methodology() {
  return (
    <div className="flex-1 overflow-y-auto min-h-0">
      <div className="max-w-[900px] mx-auto p-4 space-y-4">
        <div>
          <h2 className="font-bold text-slate-900 dark:text-slate-100 text-xl">Methodology &amp; System Overview</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">SIH 2026 | NTRO | Thermal Hotspot Classification System</p>
        </div>

        <Card title="Architecture">
          <ArchitectureDiagram />
        </Card>

        <Card title="How It Works">
          <ol className="list-decimal pl-5 space-y-1">
            <li><strong>Step 1 — Satellite Detection:</strong> NASA FIRMS VIIRS sensors on Suomi NPP and NOAA-21 detect thermal anomalies. Each detection carries latitude, longitude, FRP (fire radiative power in MW), brightness temperature (Kelvin), confidence, day/night flag, and acquisition date.</li>
            <li><strong>Step 2 — Spatial Context:</strong> For each detection, nearby OpenStreetMap infrastructure is queried within a 15 km radius. Haversine distances are computed to the nearest industry, refinery, oil/gas, mining, forest, agriculture, and power plant site. This answers: what is near this hotspot?</li>
            <li><strong>Step 3 — Feature Engineering:</strong> Thermal features (FRP, brightness, confidence) plus spatial distances (7 distances) plus density counts (industrial sites within 3 km/5 km) plus flags (day/night, satellite) are combined into a 17-feature vector.</li>
            <li><strong>Step 4 — Labeling Functions:</strong> 14 domain-expert rule classifiers. Each checks specific conditions and votes for one class or abstains. Example: FRP over 5 MW with industry inside 20 km votes industrial_fire. Rules are transparent and conservative.</li>
            <li><strong>Step 5 — XGBoost ML:</strong> A machine learning model trained on the 17 features predicts a probability distribution across all 7 classes. It captures non-linear relationships and feature interactions the rules miss.</li>
            <li><strong>Step 6 — 5-Case Fusion:</strong> LF votes and ML probabilities are combined through a 5-case decision policy (see below), producing a final classification with confidence, decision source, and a human-review flag.</li>
            <li><strong>Step 7 — Output:</strong> A GIS-ready record with full explainability: label, confidence, evidence bullets, LF vote breakdown, ML probability distribution, spatial context, and review flag.</li>
          </ol>
        </Card>

        <Card title="Labeling Functions (14 rules)">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-slate-500 dark:text-slate-400">
                  <th className="py-1 pr-3">Rule</th>
                  <th className="py-1 pr-3">Votes</th>
                  <th className="py-1 pr-3">Checks</th>
                  <th className="py-1">Threshold</th>
                </tr>
              </thead>
              <tbody>
                {ALL_LF_NAMES.map((n) => {
                  const d = LF_DESCRIPTIONS[n] ?? { votes: '—', evidence: '—', threshold: '—' };
                  return (
                    <tr key={n} className="border-t border-slate-100 dark:border-slate-700">
                      <td className="py-1 pr-3 font-mono">{n}</td>
                      <td className="py-1 pr-3">{d.votes}</td>
                      <td className="py-1 pr-3">{d.evidence}</td>
                      <td className="py-1">{d.threshold}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Classes: {ORDER.map((c) => CLASS_DISPLAY_NAMES[c]).join(' · ')}. Rules abstain on missing or weak evidence — abstention is a valid, common outcome.
          </p>
        </Card>

        <Card title="5-Case Fusion Logic">
          <p>
            Rules are transparent with hard constraints but sparse; ML is adaptive and captures
            interactions but is a black box. The hybrid policy takes the best of both: structured
            decisions where evidence agrees, and mandatory human review where it disagrees.
          </p>
          <ul className="space-y-1.5">
            {FUSION_CASES.map((c) => (
              <li key={c.name}>
                <strong>{c.name}:</strong> {c.desc}
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Limitations & Honest Boundaries">
          <ul className="list-disc pl-5 space-y-1">
            <li><strong>Data leakage and accuracy:</strong> the 14 LFs evaluate the same spatial and thermal features fed into the XGBoost model, so reported training accuracy (~99%) is inflated. Robust real-world generalization is estimated around 70–80%. Ground truth collection is needed for proper validation.</li>
            <li><strong>High abstention (~92% unclassified):</strong> the vast majority of FIRMS hotspots lack nearby OSM infrastructure within range (remote forest, grassland, sparsely mapped areas). Rather than guessing, the system abstains and flags them for human review — safe, trustworthy behavior, but it limits autonomous scale.</li>
            <li><strong>Class representation:</strong> 5 of 7 classes have rich training samples. Mining Activity and Industrial Process Heat rely on sparse or synthetic regional validation sets (1–7 samples) due to OSM tagging limits.</li>
            <li><strong>Dataset scope:</strong> this demo indexes 1,268 hotspots, backed by OSM data from 23 Indian states with about 20,231 industrial and 3,791 forest/agriculture sites.</li>
            <li><strong>Satellite limits:</strong> VIIRS revisit is roughly 12-hourly, not real-time. Cloud cover blocks thermal detection (a major monsoon-season limitation). A thermal anomaly is not a confirmed fire — it can be industrial heat, a gas flare, or a false alarm.</li>
            <li><strong>Process heat vs fire:</strong> without temporal persistence data, a normal factory furnace looks thermally similar to a factory fire. Human review matters most here.</li>
            <li><strong>OSM quality:</strong> crowd-sourced and sometimes incomplete or wrong, which directly bounds spatial-context quality.</li>
          </ul>
        </Card>

        <Card title="Data Sources">
          <ul className="list-disc pl-5 space-y-1">
            <li><strong>NASA FIRMS</strong> (Fire Information for Resource Management System): VIIRS sensors on Suomi NPP and NOAA-21. Free. Each detection provides latitude, longitude, FRP, brightness, confidence, day/night, acquisition date, and satellite.</li>
            <li><strong>OpenStreetMap</strong>: free, crowd-sourced map data under the ODbL license. Provides industry, refinery, oil/gas, mining, forest, agriculture, and power-plant infrastructure across 23 Indian states (about 20,231 industrial sites, 3,791 forest/agriculture sites). Coverage is strong in urban/industrial areas and sparse in remote regions.</li>
            <li>No paid data sources anywhere in the pipeline. Entirely free data.</li>
          </ul>
        </Card>

        <Card title="Future Scope">
          <ul className="list-disc pl-5 space-y-1">
            <li><strong>Ground truth collection:</strong> manual verification by domain experts for proper accuracy validation.</li>
            <li><strong>More data:</strong> months/years of FIRMS history across seasons (including winter stubble burning), all Indian states and UTs.</li>
            <li><strong>Temporal persistence:</strong> separate new fires from persistent industrial heat sources over time.</li>
            <li><strong>Weather integration:</strong> risk scoring from wind, humidity, temperature, and dryness.</li>
            <li><strong>Production deployment:</strong> cloud infrastructure, PostGIS database, authentication, monitoring, model registry, alerting.</li>
            <li><strong>Advanced ML:</strong> ensembles, temporal models, uncertainty estimation, active and online learning.</li>
            <li><strong>Multi-source fusion:</strong> satellite plus weather, GIS, ground reports, and human observations.</li>
            <li><strong>Predictive risk modeling and national scale:</strong> forecast likely fires before ignition, real-time processing across India.</li>
          </ul>
        </Card>
      </div>
    </div>
  );
}
