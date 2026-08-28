import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')


# ============================================
# ATTACK TECHNIQUE CLASSIFICATION
# ============================================
def classify_attack_technique(top_features):
    """
    Classify attack technique based on SHAP top features.
    Supports all attack types from CICIDS2017.
    
    Args:
        top_features: List of dicts with 'feature' and 'value' keys
    
    Returns:
        str: MITRE technique ID
    """
    technique = 'T1499'
    
    feature_dict = {f.get('feature', ''): f.get('value', 0) for f in top_features}
    feature_names = list(feature_dict.keys())
    
    # DoS Detection
    if 'FlowDuration' in feature_dict and feature_dict['FlowDuration'] > 30_000_000:
        if 'TotalLengthofBwdPackets' in feature_dict and feature_dict['TotalLengthofBwdPackets'] > 5000:
            if 'TotalFwdPackets' in feature_dict and feature_dict['TotalFwdPackets'] < 20:
                return 'T1499'
    
    # DDoS Detection
    if 'FlowPacketsPerSecond' in feature_dict and feature_dict['FlowPacketsPerSecond'] > 100:
        if 'TotalFwdPackets' in feature_dict and feature_dict['TotalFwdPackets'] > 50:
            if 'FlowDuration' in feature_dict and feature_dict['FlowDuration'] < 10_000_000:
                return 'T1498'
    
    # SYN Flood
    if 'SYNFlagCount' in feature_dict and feature_dict['SYNFlagCount'] > 10:
        if 'TotalFwdPackets' in feature_dict and feature_dict['TotalFwdPackets'] > 20:
            return 'T1498.001'
    
    # UDP Flood
    if 'Protocol' in feature_dict and feature_dict['Protocol'] == 17:
        if 'TotalFwdPackets' in feature_dict and feature_dict['TotalFwdPackets'] > 50:
            return 'T1498.001'
    
    # Port Scan
    if 'SYNFlagCount' in feature_dict and feature_dict['SYNFlagCount'] > 20:
        if 'TotalBackwardPackets' in feature_dict and feature_dict['TotalBackwardPackets'] < 10:
            if 'DestinationPort' in feature_dict and feature_dict['DestinationPort'] in [21, 22, 23, 25, 53, 80, 443, 445, 3389, 8080]:
                return 'T1046'
    
    # Brute Force
    if 'SYNFlagCount' in feature_dict and feature_dict['SYNFlagCount'] > 30:
        if 'ACKFlagCount' in feature_dict and feature_dict['ACKFlagCount'] > 30:
            if 'DestinationPort' in feature_dict and feature_dict['DestinationPort'] in [21, 22, 23, 3389]:
                return 'T1110'
    
    # Web Attacks
    if 'DestinationPort' in feature_dict and feature_dict['DestinationPort'] in [80, 443, 8080]:
        if 'PacketLengthVariance' in feature_dict and feature_dict['PacketLengthVariance'] > 10000:
            if 'BwdPacketLengthStd' in feature_dict and feature_dict['BwdPacketLengthStd'] > 100:
                return 'T1190'
    
    # Botnet / C2
    if 'DestinationPort' in feature_dict and feature_dict['DestinationPort'] in [53, 123, 161]:
        if 'TotalFwdPackets' in feature_dict and feature_dict['TotalFwdPackets'] > 20:
            return 'T1071'
    
    # Heartbleed
    if 'DestinationPort' in feature_dict and feature_dict['DestinationPort'] in [443, 8443]:
        if 'PacketLengthMean' in feature_dict and feature_dict['PacketLengthMean'] > 500:
            if 'PacketLengthStd' in feature_dict and feature_dict['PacketLengthStd'] < 10:
                return 'T1190'
    
    # Fallback: Port-based detection
    if 'DestinationPort' in feature_dict:
        port = feature_dict['DestinationPort']
        if port in [21, 22, 23, 3389]:
            return 'T1110'
        elif port in [80, 443, 8080]:
            return 'T1190'
        elif port in [53, 123, 161]:
            return 'T1071'
        elif port in [445, 139]:
            return 'T1046'
    
    return 'T1499'


# ============================================
# ATTACK TYPE TO MITRE MAPPING
# ============================================
def get_mitre_from_attack_type(attack_type):
    """Map attack type name to MITRE technique ID."""
    mapping = {
        'DoS Hulk': 'T1499',
        'DoS GoldenEye': 'T1499',
        'DoS slowloris': 'T1499',
        'DoS Slowhttptest': 'T1499',
        'DDoS': 'T1498',
        'PortScan': 'T1046',
        'FTP-Patator': 'T1110',
        'SSH-Patator': 'T1110',
        'Web Attack Brute Force': 'T1190',
        'Web Attack XSS': 'T1190',
        'Web Attack Sql Injection': 'T1190',
        'Bot': 'T1071',
        'Infiltration': 'T1071',
        'Heartbleed': 'T1190',
    }
    return mapping.get(attack_type, 'T1499')


# ============================================
# BEHAVIOR TO MITRE MAPPING
# ============================================
def map_to_mitre(behavior_pattern):
    """Map behavior patterns to MITRE ATT&CK technique IDs."""
    mappings = {
        'ddos': 'T1498',
        'dos': 'T1499',
        'flood': 'T1498.001',
        'amplification': 'T1498.002',
        'syn_flood': 'T1498.001',
        'udp_flood': 'T1498.001',
        'icmp_flood': 'T1498.001',
        'http_flood': 'T1499.002',
        'slowloris': 'T1499.002',
        'resource_exhaustion': 'T1499',
        'hulk': 'T1499',
        'goldeneye': 'T1499',
        'slowhttptest': 'T1499',
        'port_scan': 'T1046',
        'portsweep': 'T1046',
        'network_scan': 'T1046',
        'service_discovery': 'T1046',
        'brute_force': 'T1110',
        'password_brute': 'T1110',
        'ssh_brute': 'T1110',
        'ftp_brute': 'T1110',
        'patator': 'T1110',
        'ssh-patator': 'T1110',
        'ftp-patator': 'T1110',
        'web_attack': 'T1190',
        'sql_injection': 'T1190',
        'xss': 'T1190',
        'web_exploit': 'T1190',
        'brute_force_web': 'T1190',
        'bot': 'T1071',
        'c2': 'T1071',
        'infiltration': 'T1071',
        'heartbleed': 'T1190',
        'attack': 'T1499'
    }

    for pattern, tech_id in mappings.items():
        if pattern.lower() in behavior_pattern.lower():
            return tech_id

    return 'T1499'


# ============================================
# TECHNIQUES LOOKUP
# ============================================
_techniques_cache = None


def get_techniques():
    """Lazy load MITRE techniques from the CTI data file."""
    global _techniques_cache
    
    if _techniques_cache is not None:
        return _techniques_cache
    
    _techniques_cache = {}
    mitre_file = 'mitre_data/cti/enterprise-attack/enterprise-attack.json'

    if not os.path.exists(mitre_file):
        import glob
        mitre_files = glob.glob('mitre_data/**/enterprise-attack.json', recursive=True)
        if mitre_files:
            mitre_file = mitre_files[0]
        else:
            print("   ⚠️ MITRE data not found. Run: cd mitre_data && git clone https://github.com/mitre/cti.git")
            return _techniques_cache

    try:
        with open(mitre_file, 'r', encoding='utf-8') as f:
            mitre_data = json.load(f)

        for obj in mitre_data['objects']:
            if obj['type'] != 'attack-pattern':
                continue

            if obj.get('x_mitre_deprecated', False) or obj.get('revoked', False):
                continue

            tech_id = ''
            for ref in obj.get('external_references', []):
                if ref.get('source_name') == 'mitre-attack':
                    tech_id = ref.get('external_id', '')
                    break

            if tech_id:
                tactics = []
                for phase in obj.get('kill_chain_phases', []):
                    if isinstance(phase, dict):
                        if 'phase_name' in phase:
                            tactics.append(phase['phase_name'])
                        elif 'kill_chain_name' in phase:
                            tactics.append(phase['kill_chain_name'])
                    elif isinstance(phase, str):
                        tactics.append(phase)

                _techniques_cache[tech_id] = {
                    'id': tech_id,
                    'name': obj.get('name', 'Unknown'),
                    'description': obj.get('description', '')[:200] + '...' if obj.get('description') else '',
                    'tactic': ', '.join(tactics) if tactics else 'Unknown'
                }
    except Exception as e:
        print(f"   ⚠️ Error loading MITRE techniques: {e}")
    
    return _techniques_cache


techniques = get_techniques()


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("MITRE ATT&CK MAPPING")
    print("=" * 60)

    print("\n📂 Loading MITRE ATT&CK data...")
    
    techs = get_techniques()
    print(f"   ✅ Loaded {len(techs)} current techniques")
    print(f"   ℹ️  Skipped deprecated/revoked techniques")

    print("\n📝 Mapping rules defined (all attack types)")
    print("\n📊 Sample mappings:")

    sample_behaviors = [
        'ddos', 'dos', 'syn_flood', 'udp_flood', 'amplification',
        'http_flood', 'slowloris', 'hulk', 'goldeneye',
        'port_scan', 'network_scan',
        'brute_force', 'ssh_brute', 'ftp_brute',
        'web_attack', 'sql_injection', 'xss',
        'bot', 'infiltration',
        'heartbleed'
    ]

    print("\n   Behavior Pattern → MITRE Technique:")
    print("   " + "-" * 55)
    for behavior in sample_behaviors:
        tech_id = map_to_mitre(behavior)
        tech_info = techs.get(tech_id, {'name': 'Unknown', 'tactic': 'Unknown'})
        print(f"   {behavior:<20} → {tech_id} ({tech_info['name']})")

    print("\n💾 Saving mapping results...")

    os.makedirs('models', exist_ok=True)

    mapping_df = pd.DataFrame([
        {'behavior': behavior, 'technique': tech_id}
        for behavior, tech_id in [
            ('ddos', 'T1498'),
            ('dos', 'T1499'),
            ('flood', 'T1498.001'),
            ('amplification', 'T1498.002'),
            ('syn_flood', 'T1498.001'),
            ('udp_flood', 'T1498.001'),
            ('icmp_flood', 'T1498.001'),
            ('http_flood', 'T1499.002'),
            ('slowloris', 'T1499.002'),
            ('hulk', 'T1499'),
            ('goldeneye', 'T1499'),
            ('slowhttptest', 'T1499'),
            ('resource_exhaustion', 'T1499'),
            ('port_scan', 'T1046'),
            ('network_scan', 'T1046'),
            ('service_discovery', 'T1046'),
            ('brute_force', 'T1110'),
            ('ssh_brute', 'T1110'),
            ('ftp_brute', 'T1110'),
            ('patator', 'T1110'),
            ('web_attack', 'T1190'),
            ('sql_injection', 'T1190'),
            ('xss', 'T1190'),
            ('web_exploit', 'T1190'),
            ('bot', 'T1071'),
            ('c2', 'T1071'),
            ('infiltration', 'T1071'),
            ('heartbleed', 'T1190'),
        ]
    ])

    mapping_df.to_csv('models/mitre_mapping_full.csv', index=False)
    print("   ✅ Saved: mitre_mapping_full.csv")

    techniques_df = pd.DataFrame([
        {'technique_id': k, 'name': v['name'], 'tactic': v['tactic']}
        for k, v in techs.items()
    ])
    techniques_df.to_csv('models/mitre_techniques_full.csv', index=False)
    print("   ✅ Saved: mitre_techniques_full.csv")

    print("\n" + "=" * 60)
    print("MITRE MAPPING SUMMARY")
    print("=" * 60)
    print(f"   Total techniques loaded: {len(techs)}")
    print(f"   Mapping rules defined: {len(mapping_df)}")

    print("\n" + "=" * 60)
    print("✅ MITRE mapping complete!")
    print("=" * 60)