export interface SpatialContext {
  industry: number | null;
  refinery: number | null;
  oil_gas: number | null;
  mining: number | null;
  forest: number | null;
  agriculture: number | null;
  power_plant: number | null;
}
export interface RuleEngine {
  active_votes: number;
  total_lfs: number;
  active_lfs: { name: string; vote: string }[];
}
export interface MlEngine {
  predicted_class: string;
  probabilities: Record<string, number>;
}
export interface Classification {
  class: string;
  display_name: string;
  confidence: number;
  confidence_level: 'High' | 'Medium' | 'Low';
  decision_source: string;
  risk_level: string;
  requires_human_review: boolean;
  review_reason: string | null;
  explanation: string[];
  /** Rules-vs-ML disagreement. Optional: absent means no recorded conflict. */
  conflict?: boolean;
}
export interface Hotspot {
  id: string;
  lat: number;
  lon: number;
  frp: number;
  bright_ti4: number;
  confidence: 'high' | 'nominal' | 'low';
  daynight: 'D' | 'N';
  acq_date: string;
  satellite: string;
  classification: Classification;
  spatial_context: SpatialContext;
  rule_engine: RuleEngine;
  ml_engine: MlEngine;
}
export type ClassKey = 'industrial_fire' | 'gas_flare' | 'mining_activity' | 'agricultural_burn' | 'forest_natural_fire' | 'industrial_process_heat' | 'unclassified';
export const CLASS_COLORS: Record<ClassKey, string> = {
  industrial_fire: '#e53935',
  gas_flare: '#fb8c00',
  mining_activity: '#fdd835',
  agricultural_burn: '#8d6e63',
  forest_natural_fire: '#b71c1c',
  industrial_process_heat: '#8e24aa',
  unclassified: '#9e9e9e',
};
export const CLASS_DISPLAY_NAMES: Record<ClassKey, string> = {
  industrial_fire: 'Industrial Fire',
  gas_flare: 'Gas Flare',
  mining_activity: 'Mining Activity',
  agricultural_burn: 'Agricultural Burn',
  forest_natural_fire: 'Forest Fire',
  industrial_process_heat: 'Industrial Process Heat',
  unclassified: 'Unclassified — Needs Review',
};
export const ALL_LF_NAMES: string[] = [
  'lf_industry_high_frp','lf_factory_proximity_thermal','lf_industrial_zone_cluster',
  'lf_oil_gas_flare','lf_refinery_flare','lf_mining_thermal_activity','lf_mining_high_confidence',
  'lf_agriculture_vegetation_fire','lf_agriculture_burn_context','lf_forest_vegetation_fire',
  'lf_strong_forest_fire','lf_nighttime_process_heat','lf_static_industrial_heat','lf_power_plant_process_heat'
];
