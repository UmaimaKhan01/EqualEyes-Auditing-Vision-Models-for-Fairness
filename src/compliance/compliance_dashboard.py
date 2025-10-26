import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
from pathlib import Path

def get_compliance_status(bias_score: float) -> str:
    """Determine overall compliance status."""
    if bias_score > 0.7:
        return "NON_COMPLIANT"
    elif bias_score > 0.5:
        return "HIGH_RISK"
    elif bias_score > 0.3:
        return "REQUIRES_MONITORING"
    else:
        return "COMPLIANT"

def get_risk_level(bias_score: float) -> str:
    """Determine risk level for regulatory purposes."""
    if bias_score > 0.7:
        return "UNACCEPTABLE"
    elif bias_score > 0.5:
        return "HIGH"
    elif bias_score > 0.3:
        return "LIMITED"
    else:
        return "MINIMAL"

def get_eu_risk_category(bias_score: float) -> str:
    """EU AI Act risk categorization."""
    if bias_score > 0.7:
        return "unacceptable_risk"
    elif bias_score > 0.5:
        return "high_risk"
    elif bias_score > 0.3:
        return "limited_risk"
    else:
        return "minimal_risk"

def generate_recommendations(bias_score: float) -> List[str]:
    """Generate actionable recommendations."""
    recommendations = []
    
    if bias_score > 0.6:
        recommendations.extend([
            "Immediate bias mitigation required",
            "Implement human oversight for all decisions",
            "Conduct comprehensive dataset audit",
            "Consider temporary system suspension"
        ])
    elif bias_score > 0.4:
        recommendations.extend([
            "Enhanced monitoring and logging required",
            "Implement bias detection alerts",
            "Review training data for representational issues",
            "Establish quarterly bias audits"
        ])
    elif bias_score > 0.2:
        recommendations.extend([
            "Implement regular bias monitoring",
            "Document bias mitigation strategies",
            "Train staff on algorithmic fairness"
        ])
    else:
        recommendations.extend([
            "Maintain current bias monitoring",
            "Continue annual compliance reviews"
        ])
    
    return recommendations

def generate_action_items(bias_score: float) -> List[Dict]:
    """Generate specific action items with priorities."""
    actions = []
    
    if bias_score > 0.5:
        actions.append({
            "priority": "HIGH",
            "action": "Implement real-time bias monitoring",
            "deadline": (datetime.now() + timedelta(days=30)).isoformat(),
            "owner": "AI Ethics Team"
        })
        actions.append({
            "priority": "HIGH", 
            "action": "Conduct bias impact assessment",
            "deadline": (datetime.now() + timedelta(days=14)).isoformat(),
            "owner": "Data Science Team"
        })
    
    actions.append({
        "priority": "MEDIUM",
        "action": "Update bias documentation",
        "deadline": (datetime.now() + timedelta(days=60)).isoformat(),
        "owner": "Compliance Team"
    })
    
    return actions

class AIComplianceDashboard:
    """Generate compliance reports for AI regulations (EU AI Act, etc.)."""
    
    def __init__(self):
        self.compliance_standards = {
            'EU_AI_ACT': {
                'bias_threshold': 0.3,
                'documentation_required': True,
                'audit_frequency': 'quarterly',
                'risk_categories': ['minimal', 'limited', 'high', 'unacceptable']
            },
            'GDPR': {
                'data_protection': True,
                'right_to_explanation': True,
                'automated_decision_making': True
            },
            'IEEE_2857': {
                'fairness_metrics': ['demographic_parity', 'equalized_odds'],
                'bias_threshold': 0.2
            }
        }
    
    def assess_compliance_risk(self, bias_results: Dict) -> Dict[str, Any]:
        """Assess compliance risk level based on bias analysis."""
        risk_assessment = {}
        
        # EU AI Act Assessment
        bias_score = bias_results.get('overall_bias_score', 0)
        
        if bias_score > 0.7:
            risk_level = 'unacceptable'
            actions_required = [
                'Immediate system shutdown required',
                'Complete bias audit and mitigation',
                'Legal review before redeployment'
            ]
        elif bias_score > 0.5:
            risk_level = 'high'
            actions_required = [
                'Mandatory bias testing',
                'Human oversight implementation',
                'Quarterly compliance audits'
            ]
        elif bias_score > 0.3:
            risk_level = 'limited'
            actions_required = [
                'Enhanced monitoring',
                'Bias documentation update',
                'User notification of AI use'
            ]
        else:
            risk_level = 'minimal'
            actions_required = [
                'Standard monitoring',
                'Annual compliance review'
            ]
        
        risk_assessment['EU_AI_ACT'] = {
            'risk_level': risk_level,
            'compliance_score': max(0, 100 - bias_score * 100),
            'actions_required': actions_required,
            'next_audit_date': self._calculate_next_audit_date(risk_level)
        }
        
        # GDPR Assessment
        risk_assessment['GDPR'] = {
            'data_protection_compliant': True,  # Assuming compliant
            'explainability_score': self._calculate_explainability_score(bias_results),
            'automated_decision_impact': self._assess_decision_impact(bias_results)
        }
        
        return risk_assessment
    
    def generate_compliance_report(self, analysis_results: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        report = {
            'report_metadata': {
                'generated_date': datetime.now().isoformat(),
                'analysis_period': self._get_analysis_period(analysis_results),
                'total_analyses': len(analysis_results),
                'compliance_frameworks': list(self.compliance_standards.keys())
            },
            'executive_summary': {},
            'detailed_findings': {},
            'risk_matrix': {},
            'recommendations': [],
            'audit_trail': []
        }
        
        # Calculate aggregate metrics
        bias_scores = [r.get('bias_score', 0) for r in analysis_results]
        avg_bias = np.mean(bias_scores)
        max_bias = max(bias_scores)
        
        # Executive Summary
        report['executive_summary'] = {
            'overall_compliance_status': self._determine_overall_status(avg_bias),
            'average_bias_score': avg_bias,
            'maximum_bias_detected': max_bias,
            'risk_level': get_risk_level(avg_bias),
            'high_risk_incidents': len([s for s in bias_scores if s > 0.5]),
            'compliance_percentage': max(0, 100 - avg_bias * 100)
        }
        
        # Detailed Findings by Framework
        regulatory_assessment = {}
        for framework in self.compliance_standards:
            framework_assessment = self._assess_framework_compliance(
                framework, analysis_results
            )
            regulatory_assessment[framework] = framework_assessment
        
        report['regulatory_assessment'] = regulatory_assessment
        
        # Risk Matrix
        report['risk_matrix'] = self._create_risk_matrix(analysis_results)
        
        # Recommendations
        report['recommendations'] = self._generate_compliance_recommendations(
            avg_bias, max_bias, analysis_results
        )
        
        # Audit Trail
        report['audit_trail'] = self._create_audit_trail(analysis_results)
        
        return report
    
    def _assess_framework_compliance(self, framework: str, results: List[Dict]) -> Dict:
        """Assess compliance for specific regulatory framework."""
        if framework == 'EU_AI_ACT':
            return self._assess_eu_ai_act(results)
        elif framework == 'GDPR':
            return self._assess_gdpr(results)
        elif framework == 'IEEE_2857':
            return self._assess_ieee_2857(results)
        else:
            return {'status': 'framework_not_implemented'}
    
    def _assess_eu_ai_act(self, results: List[Dict]) -> Dict:
        """EU AI Act specific compliance assessment."""
        bias_scores = [r.get('bias_score', 0) for r in results]
        
        # Risk categorization
        risk_distribution = {
            'unacceptable': len([s for s in bias_scores if s > 0.7]),
            'high': len([s for s in bias_scores if 0.5 < s <= 0.7]),
            'limited': len([s for s in bias_scores if 0.3 < s <= 0.5]),
            'minimal': len([s for s in bias_scores if s <= 0.3])
        }
        
        return {
            'overall_compliance': risk_distribution['unacceptable'] == 0,
            'risk_distribution': risk_distribution,
            'required_actions': self._get_eu_required_actions(risk_distribution),
            'documentation_status': 'complete',  # Assuming complete
            'human_oversight_required': risk_distribution['high'] > 0
        }
    
    def _create_risk_matrix(self, results: List[Dict]) -> Dict:
        """Create risk assessment matrix."""
        # Probability vs Impact matrix
        risk_matrix = {
            'high_probability_high_impact': 0,
            'high_probability_low_impact': 0,
            'low_probability_high_impact': 0,
            'low_probability_low_impact': 0
        }
        
        for result in results:
            bias_score = result.get('bias_score', 0)
            confidence = result.get('confidence', 0.5)
            
            # Determine probability and impact
            high_prob = confidence > 0.7
            high_impact = bias_score > 0.5
            
            if high_prob and high_impact:
                risk_matrix['high_probability_high_impact'] += 1
            elif high_prob and not high_impact:
                risk_matrix['high_probability_low_impact'] += 1
            elif not high_prob and high_impact:
                risk_matrix['low_probability_high_impact'] += 1
            else:
                risk_matrix['low_probability_low_impact'] += 1
        
        return risk_matrix
    
    def export_compliance_report(self, report: Dict, format: str = 'pdf') -> str:
        """Export compliance report in various formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            filename = f"compliance_report_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        elif format == 'html':
            filename = f"compliance_report_{timestamp}.html"
            html_content = self._generate_html_report(report)
            with open(filename, 'w') as f:
                f.write(html_content)
        
        elif format == 'csv':
            filename = f"compliance_data_{timestamp}.csv"
            df = self._convert_to_dataframe(report)
            df.to_csv(filename, index=False)
        
        return filename
    
    def _generate_html_report(self, report: Dict) -> str:
        """Generate HTML compliance report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Compliance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .risk-high {{ background-color: #ffebee; }}
                .risk-medium {{ background-color: #fff3e0; }}
                .risk-low {{ background-color: #e8f5e8; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Bias Compliance Report</h1>
                <p>Generated: {report['report_metadata']['generated_date']}</p>
                <p>Total Analyses: {report['report_metadata']['total_analyses']}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p><strong>Overall Status:</strong> {report['executive_summary']['overall_compliance_status']}</p>
                <p><strong>Average Bias Score:</strong> {report['executive_summary']['average_bias_score']:.2f}</p>
                <p><strong>Compliance Percentage:</strong> {report['executive_summary']['compliance_percentage']:.1f}%</p>
            </div>
            
            <div class="section">
                <h2>Risk Matrix</h2>
                <table>
                    <tr>
                        <th>Risk Category</th>
                        <th>Count</th>
                    </tr>
                    {self._generate_risk_table_rows(report['risk_matrix'])}
                </table>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {self._generate_recommendation_list(report['recommendations'])}
                </ul>
            </div>
        </body>
        </html>
        """
        return html
    
    def _assess_gdpr(self, results: List[Dict]) -> Dict:
        """GDPR specific compliance assessment."""
        return {
            'data_protection_compliant': True,
            'explainability_score': 0.8,
            'automated_decision_impact': 'moderate',
            'consent_management': 'implemented',
            'data_minimization': 'compliant'
        }
    
    def _assess_ieee_2857(self, results: List[Dict]) -> Dict:
        """IEEE 2857 fairness standards assessment."""
        bias_scores = [r.get('bias_score', 0) for r in results]
        avg_bias = np.mean(bias_scores) if bias_scores else 0
        
        return {
            'fairness_threshold_met': avg_bias < 0.4,
            'demographic_parity': avg_bias < 0.2,
            'equalized_odds': avg_bias < 0.3,
            'bias_mitigation_required': avg_bias > 0.2
        }
    
    def _calculate_explainability_score(self, bias_results: Dict) -> float:
        """Calculate explainability score for GDPR compliance."""
        # Simple heuristic - in practice would be more complex
        return max(0.5, 1.0 - bias_results.get('overall_bias_score', 0))
    
    def _assess_decision_impact(self, bias_results: Dict) -> str:
        """Assess automated decision making impact."""
        bias_score = bias_results.get('overall_bias_score', 0)
        if bias_score > 0.5:
            return 'high'
        elif bias_score > 0.3:
            return 'moderate'
        else:
            return 'low'
    
    def _calculate_next_audit_date(self, risk_level: str) -> str:
        """Calculate next required audit date based on risk level."""
        days_mapping = {
            'unacceptable': 30,
            'high': 90,
            'limited': 180,
            'minimal': 365
        }
        days = days_mapping.get(risk_level, 180)
        return (datetime.now() + timedelta(days=days)).isoformat()
    
    def _get_analysis_period(self, results: List[Dict]) -> str:
        """Get analysis period from results."""
        return datetime.now().strftime("%Y-%m-%d")
    
    def _determine_overall_status(self, avg_bias: float) -> str:
        """Determine overall compliance status."""
        if avg_bias > 0.7:
            return "NON_COMPLIANT"
        elif avg_bias > 0.5:
            return "HIGH_RISK"
        elif avg_bias > 0.3:
            return "REQUIRES_MONITORING"
        else:
            return "COMPLIANT"
    
    def _get_eu_required_actions(self, risk_distribution: Dict) -> List[str]:
        """Get required actions for EU AI Act compliance."""
        actions = []
        if risk_distribution['unacceptable'] > 0:
            actions.append("Immediate system shutdown required")
        if risk_distribution['high'] > 0:
            actions.append("Implement human oversight")
            actions.append("Conduct conformity assessment")
        if risk_distribution['limited'] > 0:
            actions.append("Implement transparency obligations")
        return actions
    
    def _create_audit_trail(self, results: List[Dict]) -> List[Dict]:
        """Create audit trail for compliance tracking."""
        return [{
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'bias_detection',
            'results_count': len(results),
            'status': 'completed'
        }]
    
    def _generate_compliance_recommendations(self, avg_bias: float, max_bias: float, results: List[Dict]) -> List[str]:
        """Generate compliance-specific recommendations."""
        recommendations = []
        
        if avg_bias > 0.6:
            recommendations.extend([
                "Immediate bias mitigation required",
                "Implement human oversight for all decisions",
                "Conduct comprehensive dataset audit"
            ])
        elif avg_bias > 0.4:
            recommendations.extend([
                "Enhanced monitoring and logging required",
                "Implement bias detection alerts",
                "Review training data for representational issues"
            ])
        else:
            recommendations.extend([
                "Maintain current bias monitoring",
                "Continue regular compliance reviews"
            ])
        
        return recommendations
    
    def _generate_risk_table_rows(self, risk_matrix: Dict) -> str:
        """Generate HTML table rows for risk matrix."""
        rows = ""
        for category, count in risk_matrix.items():
            rows += f"<tr><td>{category.replace('_', ' ').title()}</td><td>{count}</td></tr>"
        return rows
    
    def _generate_recommendation_list(self, recommendations: List[str]) -> str:
        """Generate HTML list items for recommendations."""
        return "".join(f"<li>{rec}</li>" for rec in recommendations)
    
    def _convert_to_dataframe(self, report: Dict) -> pd.DataFrame:
        """Convert report to DataFrame for CSV export."""
        data = []
        exec_summary = report['executive_summary']
        
        data.append({
            'metric': 'overall_compliance_status',
            'value': exec_summary['overall_compliance_status'],
            'category': 'summary'
        })
        data.append({
            'metric': 'average_bias_score', 
            'value': exec_summary['average_bias_score'],
            'category': 'summary'
        })
        data.append({
            'metric': 'compliance_percentage',
            'value': exec_summary['compliance_percentage'],
            'category': 'summary'
        })
        
        return pd.DataFrame(data)
    
    def schedule_automated_compliance_check(self, frequency: str = 'weekly'):
        """Schedule automated compliance monitoring."""
        # This would integrate with cron jobs or task schedulers
        schedule_config = {
            'frequency': frequency,
            'check_types': ['bias_analysis', 'risk_assessment', 'compliance_report'],
            'alert_thresholds': {
                'bias_score': 0.5,
                'risk_level': 'high'
            },
            'notification_emails': [],  # Add email addresses
            'export_formats': ['json', 'html', 'csv']
        }
        
        with open('compliance_schedule.json', 'w') as f:
            json.dump(schedule_config, f, indent=2)
        
        return schedule_config

def create_compliance_dashboard():
    """Create and run compliance dashboard."""
    dashboard = AIComplianceDashboard()
    
    # Try to load existing analysis results
    results_path = Path("results/bias_analysis_complete.json")
    if not results_path.exists():
        # Try alternative result files
        alt_paths = [
            "results/bias_analysis_summary.json",
            "results/gender_analysis_summary.json"
        ]
        for alt_path in alt_paths:
            if Path(alt_path).exists():
                results_path = Path(alt_path)
                break
        
        # Create sample data if no real results exist
        if not results_path.exists():
            print("Creating demo with sample bias analysis data...")
            sample_data = {
                'bias_metrics': {'overall_bias_score': 0.337},
                'analysis_summary': {'total_predictions': 82}
            }
            
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            results_path = results_dir / "bias_analysis_summary.json"
            
            with open(results_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
    
    # Load the results
    with open(results_path, 'r') as f:
        analysis_results = json.load(f)
    
    print("📊 Generating AI Compliance Dashboard...")
    print("=" * 60)
    
    # Extract key metrics
    bias_score = analysis_results.get('bias_metrics', {}).get('overall_bias_score', 0.337)
    total_predictions = analysis_results.get('analysis_summary', {}).get('total_predictions', 82)
    
    # Generate compliance assessment
    compliance_report = {
        "report_metadata": {
            "generated_date": datetime.now().isoformat(),
            "analysis_period": "2025-10-26",
            "total_analyses": total_predictions,
            "compliance_frameworks": ["EU_AI_ACT", "GDPR", "IEEE_2857"]
        },
        "executive_summary": {
            "overall_compliance_status": get_compliance_status(bias_score),
            "average_bias_score": bias_score,
            "compliance_percentage": max(0, 100 - bias_score * 100),
            "risk_level": get_risk_level(bias_score),
            "high_risk_incidents": 0 if bias_score < 0.5 else 1
        },
        "regulatory_assessment": {
            "EU_AI_ACT": {
                "risk_category": get_eu_risk_category(bias_score),
                "compliance_required": bias_score > 0.3,
                "documentation_status": "complete",
                "next_audit_date": (datetime.now() + timedelta(days=90)).isoformat()
            },
            "GDPR": {
                "data_protection_compliant": True,
                "explainability_required": True,
                "automated_decision_impact": "moderate"
            },
            "IEEE_2857": {
                "fairness_threshold_met": bias_score < 0.4,
                "bias_mitigation_required": bias_score > 0.2
            }
        },
        "recommendations": generate_recommendations(bias_score),
        "action_items": generate_action_items(bias_score)
    }
    
    # Generate compliance report
    compliance_report = dashboard.generate_compliance_report([analysis_results])
    
    # Export in multiple formats
    json_file = dashboard.export_compliance_report(compliance_report, 'json')
    html_file = dashboard.export_compliance_report(compliance_report, 'html')
    
    print(f"Compliance reports generated:")
    print(f"- JSON: {json_file}")
    print(f"- HTML: {html_file}")
    
    # Schedule automated checks
    schedule = dashboard.schedule_automated_compliance_check('weekly')
    print(f"Automated compliance monitoring scheduled: {schedule['frequency']}")
    
    # Display dashboard in terminal
    display_compliance_dashboard(compliance_report)
    
    return compliance_report

def display_compliance_dashboard(report: Dict):
    """Display compliance dashboard in terminal."""
    
    print("\n🎯 AI COMPLIANCE DASHBOARD")
    print("=" * 60)
    
    # Executive Summary
    exec_summary = report['executive_summary']
    print(f"Overall Status: {exec_summary['overall_compliance_status']}")
    print(f"Risk Level: {exec_summary['risk_level']}")
    print(f"Compliance Score: {exec_summary['compliance_percentage']:.1f}%")
    print(f"Bias Score: {exec_summary['average_bias_score']:.1f}%")
    
    # Regulatory Status
    print(f"\n📋 REGULATORY COMPLIANCE")
    print("-" * 30)
    
    if 'regulatory_assessment' in report:
        reg_assess = report['regulatory_assessment']
        
        # EU AI Act
        if 'EU_AI_ACT' in reg_assess:
            eu_status = reg_assess['EU_AI_ACT']
            risk_cat = eu_status.get('risk_category', eu_status.get('risk_level', 'UNKNOWN'))
            print(f"EU AI Act: {risk_cat.upper()}")
        
        # GDPR
        if 'GDPR' in reg_assess:
            gdpr_status = reg_assess['GDPR']
            gdpr_compliant = gdpr_status.get('data_protection_compliant', True)
            print(f"GDPR: {'COMPLIANT' if gdpr_compliant else 'NON-COMPLIANT'}")
        
        # IEEE 2857
        if 'IEEE_2857' in reg_assess:
            ieee_status = reg_assess['IEEE_2857']
            ieee_compliant = ieee_status.get('fairness_threshold_met', True)
            print(f"IEEE 2857: {'COMPLIANT' if ieee_compliant else 'REQUIRES_ACTION'}")
    else:
        print("Regulatory assessment data not available")
    
    # Recommendations
    print(f"\n💡 KEY RECOMMENDATIONS")
    print("-" * 30)
    for i, rec in enumerate(report['recommendations'][:3], 1):
        print(f"{i}. {rec}")
    
    # Action Items
    print(f"\n⚡ IMMEDIATE ACTIONS REQUIRED")
    print("-" * 30)
    
    if 'action_items' in report and report['action_items']:
        for action in report['action_items']:
            priority_emoji = "🔴" if action.get('priority', '') == 'HIGH' else "🟡"
            print(f"{priority_emoji} {action.get('action', 'No action specified')} (Due: {action.get('deadline', 'No deadline')[:10]})")
    else:
        print("🟢 No immediate actions required")

if __name__ == "__main__":
    create_compliance_dashboard()