import Image from "next/image";

const metrics = [
  ["0.7840", "Final holdout ROC-AUC"],
  ["0.2732", "Final holdout PR-AUC"],
  ["0.0663", "Brier score"],
  ["113", "Automated tests"],
];

const evidence = [
  ["Discrimination", "ROC-AUC", "0.7840", "One-time holdout"],
  ["Minority-class quality", "PR-AUC", "0.2732", "One-time holdout"],
  ["Probability accuracy", "Brier score", "0.0663", "Calibrated"],
  ["Decision policy", "Recall / precision", "42.57% / 27.28%", "Threshold 0.1667"],
];

const stages = [
  ["01", "Validate", "Contract checks and relational coverage across Home Credit tables."],
  ["02", "Engineer", "Application attributes plus 196 leakage-safe historical aggregates."],
  ["03", "Model", "Benchmarks, XGBoost selection, calibration and cost-sensitive policy."],
  ["04", "Govern", "SHAP, subgroup diagnostics, model card and permanently sealed holdout."],
  ["05", "Serve", "FastAPI, Streamlit, Docker, monitoring and evidence-grounded assistant."],
];

export default function Home() {
  return (
    <main>
      <nav className="nav shell" aria-label="Primary navigation">
        <a className="brand" href="#top" aria-label="RiskLens AI home">
          <span className="brand-mark">R</span><span>RiskLens AI</span>
        </a>
        <div className="nav-links">
          <a href="#evidence">Evidence</a><a href="#workflow">Workflow</a>
          <a href="#governance">Governance</a><a href="#architecture">Architecture</a>
        </div>
        <a className="button button-small" href="https://github.com/lucifersaha02/risklens-ai" target="_blank" rel="noreferrer">View GitHub ↗</a>
      </nav>

      <header id="top" className="hero shell">
        <div className="hero-copy">
          <p className="eyebrow"><span /> Explainable credit risk intelligence</p>
          <h1>Credit risk,<br/><em>made accountable.</em></h1>
          <p className="lede">An end-to-end data science system that turns Home Credit data into calibrated risk evidence—then makes every prediction traceable, governed and ready for human review.</p>
          <div className="hero-actions">
            <a className="button" href="#evidence">Explore the evidence</a>
            <a className="text-link" href="https://github.com/lucifersaha02/risklens-ai/blob/main/reports/model_card.md" target="_blank" rel="noreferrer">Read the model card <span>↗</span></a>
          </div>
        </div>
        <div className="lens-card" aria-label="RiskLens governance summary">
          <div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="orbit orbit-three"/>
          <div className="lens-center"><strong>0.784</strong><span>ROC-AUC</span></div>
          <div className="signal signal-a"><span>01</span> calibrated probability</div>
          <div className="signal signal-b"><span>02</span> local reason codes</div>
          <div className="signal signal-c"><span>03</span> human decision</div>
        </div>
      </header>

      <section className="metrics shell" aria-label="Project metrics">
        {metrics.map(([value,label]) => <div className="metric" key={label}><strong>{value}</strong><span>{label}</span></div>)}
      </section>

      <section className="section shell intro" id="evidence">
        <div><p className="eyebrow"><span /> The portfolio case</p><h2>Beyond a good model score.</h2></div>
        <div className="intro-copy"><p>A credible lending model is more than an algorithm. RiskLens documents the data, split strategy, calibration, operating threshold, subgroup behavior, explanations and post-deployment drift.</p><p className="note">Research decision support only. It does not autonomously approve or decline loans.</p></div>
      </section>

      <section className="section shell evidence-grid">
        <div className="evidence-panel">
          <div className="panel-heading"><p className="kicker">Frozen portfolio evidence</p><span className="status">● model locked</span></div>
          <div className="table-wrap"><table><thead><tr><th>Dimension</th><th>Measure</th><th>Result</th><th>Evidence</th></tr></thead><tbody>
            {evidence.map(row => <tr key={row[0]}>{row.map(cell => <td key={cell}>{cell}</td>)}</tr>)}
          </tbody></table></div>
          <p className="table-foot">Evaluated once on 30,752 untouched applicants. Post-holdout tuning is prohibited.</p>
        </div>
        <aside className="policy-card"><p className="kicker">Locked operating policy</p><strong>16.67%</strong><p>Applications above this calibrated risk threshold enter enhanced human review.</p><div className="policy-line"><span>Hypothetical cost ratio</span><b>5 : 1</b></div><small>False-negative cost : false-positive cost. This is a documented portfolio assumption, not a lender estimate.</small></aside>
      </section>

      <section className="section full-band" id="workflow"><div className="shell">
        <div className="section-heading"><div><p className="eyebrow"><span /> Two assessment modes</p><h2>One platform. Two honest contexts.</h2></div><p>RiskLens never pretends that a brand-new application has the same evidence as an applicant already represented in historical records.</p></div>
        <div className="mode-grid">
          <article className="mode-card"><div className="mode-number">01</div><p className="kicker">Existing applicant</p><h3>Full-history assessment</h3><p>Scores an applicant from the Kaggle population using application data plus bureau, prior application, installment, credit-card and POS history.</p><ul><li>688 transformed governed features</li><li>Applicant-specific SHAP drivers</li><li>Frozen calibrated model</li></ul><Image src="/risklens-ai/existing-applicant.png" width={1600} height={900} alt="RiskLens existing applicant assessment dashboard"/></article>
          <article className="mode-card coral"><div className="mode-number">02</div><p className="kicker">New application</p><h3>Application-time simulator</h3><p>Accepts only information available at application time and uses a separately evaluated application-only model. It estimates risk; it does not grant a loan.</p><ul><li>Input range and ratio checks</li><li>Transparent application context</li><li>Direct sensitive fields excluded</li></ul><Image src="/risklens-ai/new-application-simulator.png" width={1600} height={900} alt="RiskLens new application risk simulator"/></article>
        </div>
      </div></section>

      <section className="section shell" id="architecture">
        <div className="section-heading"><div><p className="eyebrow"><span /> Reproducible workflow</p><h2>Evidence at every stage.</h2></div><p>Each stage is executable from the CLI, tested, versioned and represented by saved artifacts.</p></div>
        <div className="stages">{stages.map(([n,title,copy]) => <article key={n}><span>{n}</span><h3>{title}</h3><p>{copy}</p></article>)}</div>
        <div className="architecture"><div><p className="kicker">System architecture</p><h3>From raw records to governed decisions</h3><p>Raw Kaggle tables feed validated feature pipelines. Frozen artifacts power the authenticated API, while the dashboard presents decisions, evidence, monitoring and retrieval-grounded policy answers.</p></div><div className="architecture-flow"><span>Home Credit data</span><b>→</b><span>Feature store</span><b>→</b><span>Frozen models</span><b>→</b><span>FastAPI</span><b>→</b><span>Streamlit</span></div></div>
      </section>

      <section className="section shell" id="governance">
        <div className="section-heading"><div><p className="eyebrow"><span /> Responsible AI</p><h2>Built for scrutiny.</h2></div><p>The design surfaces limitations instead of hiding them behind a single performance number.</p></div>
        <div className="governance-grid">
          <article><span className="icon">◎</span><h3>Explainability</h3><p>Global and local SHAP analysis with business-readable labels. Explanations describe model behavior—not causality.</p></article>
          <article><span className="icon">◇</span><h3>Fairness diagnostics</h3><p>Recall, false-positive rate, selection rate and calibration are reported by gender and age band for audit only.</p></article>
          <article><span className="icon">△</span><h3>Drift monitoring</h3><p>Frozen references track prediction and feature PSI. Alerts trigger investigation, never silent post-holdout tuning.</p></article>
          <article><span className="icon">□</span><h3>Evidence assistant</h3><p>Retrieval-grounded answers cite approved project evidence. Retrieval reached 100% hit@3 on its evaluation set.</p></article>
        </div>
        <div className="limitation"><div><p className="kicker">Known limitation</p><h3>Age-band disparities remain material.</h3></div><p>Subgroup gaps are diagnostic, not proof of fairness or legal compliance. Any real deployment would require lender-specific data, validation, policy review and ongoing human oversight.</p><a href="https://github.com/lucifersaha02/risklens-ai/blob/main/reports/model_card.md" target="_blank" rel="noreferrer">Review limitations ↗</a></div>
      </section>

      <section className="section shell stack"><p className="eyebrow"><span /> Production-minded engineering</p><h2>Built to run, test and explain.</h2><div className="stack-list"><span>Python 3.12</span><span>XGBoost</span><span>scikit-learn</span><span>SHAP</span><span>FastAPI</span><span>Streamlit</span><span>Docker</span><span>GitHub Actions</span><span>Ruff</span><span>Pytest</span></div></section>

      <footer><div className="shell footer-grid"><div><div className="brand"><span className="brand-mark">R</span><span>RiskLens AI</span></div><h2>Credit risk intelligence with a complete audit trail.</h2></div><div><p>Explore the code, reports, tests and reproducible commands.</p><a className="button" href="https://github.com/lucifersaha02/risklens-ai" target="_blank" rel="noreferrer">Open the repository ↗</a></div></div><div className="shell footer-bottom"><span>© 2026 Supratik Saha</span><span>MIT licensed · Home Credit Default Risk data</span></div></footer>
    </main>
  );
}
