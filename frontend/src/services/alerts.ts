import { ClassKey, Hotspot } from '../types/hotspot';

export type Severity = 'critical' | 'high' | 'medium' | 'low';

export interface AlertAction {
  priority: 'critical' | 'high' | 'medium';
  action: string;
}

export interface AlertItem {
  /** Index into the hotspots array this alert was built from. */
  hotspotIndex: number;
  severity: Severity;
  reason: string;
  actions: AlertAction[];
}

const HIGH_RISK: ClassKey[] = ['industrial_fire', 'industrial_process_heat', 'forest_natural_fire'];
const MEDIUM_RISK: ClassKey[] = ['gas_flare', 'mining_activity', 'agricultural_burn'];

// Static mirror of backend recommendation_service.py templates (presentation
// only — these never feed back into classification).
const ACTIONS: Record<string, AlertAction[]> = {
  industrial_fire: [
    { priority: 'critical', action: 'Alert relevant industrial safety and fire authorities; inspect proximate factories/refineries.' },
    { priority: 'high', action: 'Verify the event against recent high-resolution satellite observations.' },
    { priority: 'medium', action: 'Monitor neighbouring industrial infrastructure for heat persistence over the next detection cycle.' },
  ],
  industrial_process_heat: [
    { priority: 'critical', action: 'Alert relevant industrial safety and fire authorities; inspect proximate factories/refineries.' },
    { priority: 'high', action: 'Verify the event against recent high-resolution satellite observations.' },
    { priority: 'medium', action: 'Monitor neighbouring industrial infrastructure for heat persistence over the next detection cycle.' },
  ],
  forest_natural_fire: [
    { priority: 'critical', action: 'Notify the forest department range officer; check spread toward habitations.' },
    { priority: 'high', action: 'Cross-check wind direction and dry-spell maps for escalation risk.' },
    { priority: 'medium', action: 'Track FRP trend across the next satellite passes.' },
  ],
  gas_flare: [
    { priority: 'high', action: 'Log for routine monitoring; verify flare persistence on the next detection cycle.' },
    { priority: 'medium', action: 'Confirm against known oil/gas facility flare schedules if available.' },
  ],
  mining_activity: [
    { priority: 'high', action: 'Log for routine monitoring; verify persistence on the next detection cycle.' },
    { priority: 'medium', action: 'Cross-check against active mining lease boundaries.' },
  ],
  agricultural_burn: [
    { priority: 'high', action: 'Log for routine monitoring; verify persistence on the next detection cycle.' },
    { priority: 'medium', action: 'Correlate with seasonal stubble-burning calendars.' },
  ],
  unclassified: [
    { priority: 'critical', action: 'Assign to a human analyst — do not auto-escalate.' },
    { priority: 'high', action: 'Collect OSM context or high-resolution imagery before any action.' },
  ],
};

/** Severity for one hotspot. Evaluated top-down; first match wins. */
export function severityFor(h: Hotspot): Severity {
  const cls = h.classification.class as ClassKey;
  const conf = h.classification.confidence;
  if (cls === 'unclassified') return 'critical';
  if (h.classification.conflict === true) return 'critical';
  let sev: Severity;
  if ((HIGH_RISK as string[]).includes(cls)) sev = 'high';
  else if ((MEDIUM_RISK as string[]).includes(cls)) sev = 'medium';
  else sev = 'low';
  if (conf < 0.6) {
    if (sev === 'low') sev = 'medium';
    else if (sev === 'medium') sev = 'high';
    else if (sev === 'high') sev = 'critical';
  }
  return sev;
}

function reasonFor(h: Hotspot): string {
  if (h.classification.conflict === true) {
    return 'Rules and ML disagree on this hotspot — contradictory evidence.';
  }
  if (h.classification.review_reason) return h.classification.review_reason;
  if (h.classification.class === 'unclassified') {
    return 'Insufficient evidence for any class — abstained safely.';
  }
  if (h.classification.confidence < 0.6) {
    return `Low classification confidence (${(h.classification.confidence * 100).toFixed(1)}%).`;
  }
  return `${h.classification.display_name} flagged by risk policy.`;
}

/** Build alert items for every hotspot that needs attention. */
export function buildAlerts(hotspots: Hotspot[]): AlertItem[] {
  const out: AlertItem[] = [];
  hotspots.forEach((h, i) => {
    const needsAttention =
      h.classification.requires_human_review ||
      h.classification.conflict === true ||
      h.classification.class === 'unclassified';
    if (!needsAttention) return;
    const cls = h.classification.class;
    out.push({
      hotspotIndex: i,
      severity: severityFor(h),
      reason: reasonFor(h),
      actions: ACTIONS[cls] ?? ACTIONS['unclassified'],
    });
  });
  return out;
}

export function alertCounts(alerts: AlertItem[]): Record<Severity, number> {
  const c: Record<Severity, number> = { critical: 0, high: 0, medium: 0, low: 0 };
  for (const a of alerts) c[a.severity]++;
  return c;
}
