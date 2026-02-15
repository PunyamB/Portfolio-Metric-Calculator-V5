const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, BorderStyle, WidthType, ShadingType } = require("docx");

const args = process.argv.slice(2);
const raw = JSON.parse(fs.readFileSync(args[0], "utf-8"));
const data = raw.data;
const narrative = raw.narrative;
const outputPath = args[1];

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cm = { top: 60, bottom: 60, left: 100, right: 100 };

function hCell(text, w) {
    return new TableCell({ borders, width: { size: w, type: WidthType.DXA },
        shading: { fill: "1B2A4A", type: ShadingType.CLEAR }, margins: cm,
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
            new TextRun({ text, bold: true, font: "Arial", size: 18, color: "FFFFFF" })
        ]})]
    });
}

function dCell(text, w, opts = {}) {
    return new TableCell({ borders, width: { size: w, type: WidthType.DXA },
        shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined, margins: cm,
        children: [new Paragraph({ alignment: opts.align || AlignmentType.CENTER, children: [
            new TextRun({ text: String(text), font: "Arial", size: 18, color: opts.color || "000000", bold: opts.bold || false })
        ]})]
    });
}

function secTitle(text) {
    return new Paragraph({ spacing: { before: 300, after: 120 },
        children: [new TextRun({ text, bold: true, font: "Arial", size: 24, color: "1B2A4A" })] });
}

function body(text) {
    return new Paragraph({ spacing: { after: 120 },
        children: [new TextRun({ text: text || "[No narrative generated]", font: "Arial", size: 20 })] });
}

function fmtD(v) {
    if (v == null) return "N/A";
    const n = Number(v); const p = n < 0 ? "-$" : "$";
    return p + Math.abs(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtN(v, d) { return v != null ? Number(v).toFixed(d) : "N/A"; }

let children = [];

// Title
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: `${data.report_type} Portfolio Report`, font: "Arial", size: 40, bold: true, color: "1B2A4A" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: `${data.period_start} \u2014 ${data.period_end}`, font: "Arial", size: 22, color: "666666" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
    children: [new TextRun({ text: "Bulls Before Bros | UConn MSFERM FNCE 5322", font: "Arial", size: 20, color: "888888" })] }));

// 1. Executive Summary
children.push(secTitle("1. Executive Summary"));
children.push(body(narrative.executive_summary));

// 2. Performance Snapshot
children.push(secTitle("2. Performance Snapshot"));
children.push(new Paragraph({ spacing: { after: 80 }, children: [] }));

const sc = [2400, 1740, 1740, 1740, 1740];
const metrics = [
    { l: "Portfolio Value", p: fmtD(data.start_value), c: fmtD(data.end_value), ch: fmtD(data.value_change), up: data.value_change >= 0 },
    { l: "Positions", p: String(data.start_holdings_count), c: String(data.end_holdings_count),
      ch: String(data.end_holdings_count - data.start_holdings_count), up: true },
    { l: "Trades in Period", p: "-", c: String(data.mid_period_trades), ch: "-", up: true },
];

const mKeys = ["Sharpe Ratio", "Sortino Ratio", "Beta", "Alpha (Annual)", "VaR (95%)", "Max Drawdown", "Std Dev (Annual)"];
for (const k of mKeys) {
    const p = data.start_metrics ? data.start_metrics[k] : null;
    const c = data.end_metrics ? data.end_metrics[k] : null;
    const d2 = data.metric_deltas ? data.metric_deltas[k] : null;
    metrics.push({ l: k, p: p != null ? fmtN(p, 4) : "N/A", c: c != null ? fmtN(c, 4) : "N/A",
        ch: d2 != null ? (d2 >= 0 ? "+" : "") + fmtN(d2, 4) : "N/A", up: d2 >= 0 });
}

children.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: sc,
    rows: [
        new TableRow({ children: [hCell("Metric", sc[0]), hCell("Start", sc[1]), hCell("End", sc[2]), hCell("Change", sc[3]), hCell("Signal", sc[4])] }),
        ...metrics.map(m => new TableRow({ children: [
            dCell(m.l, sc[0], { align: AlignmentType.LEFT, bold: true }),
            dCell(m.p, sc[1]), dCell(m.c, sc[2]),
            dCell(m.ch, sc[3], { color: m.up ? "007A33" : "CC0000" }),
            dCell(m.up ? "\u2191" : "\u2193", sc[4], { color: m.up ? "007A33" : "CC0000" }),
        ]}))
    ]
}));

children.push(new Paragraph({ spacing: { after: 60 }, children: [
    new TextRun({ text: `Benchmark: S&P 500 return: ${data.benchmark_return}% | Risk-free rate: ${(data.risk_free_rate * 100).toFixed(2)}%`,
        font: "Arial", size: 16, color: "888888", italics: true })
]}));

// 3. Performance Attribution
children.push(secTitle("3. Performance Attribution"));
children.push(body(narrative.performance_attribution));

const mc = [1200, 2400, 1200, 1400, 1200, 1960];
const mh = ["Ticker", "Company", "Return %", "$ Impact", "Shares", "Status"];

function moverTable(movers, color) {
    if (!movers || movers.length === 0) return new Paragraph({ children: [new TextRun({ text: "None", font: "Arial", size: 18, color: "888888" })] });
    return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: mc,
        rows: [
            new TableRow({ children: mc.map((w, i) => hCell(mh[i], w)) }),
            ...movers.map(m => new TableRow({ children: [
                dCell(m.ticker, mc[0], { bold: true }),
                dCell(m.company || m.ticker, mc[1], { align: AlignmentType.LEFT }),
                dCell((m.return_pct >= 0 ? "+" : "") + m.return_pct + "%", mc[2], { color }),
                dCell(fmtD(m.pnl), mc[3], { color }),
                dCell(String(Math.round(m.shares)), mc[4]),
                dCell(m.note || "", mc[5]),
            ]}))
        ]
    });
}

children.push(new Paragraph({ spacing: { after: 80 }, children: [
    new TextRun({ text: "Top Gainers", bold: true, font: "Arial", size: 20, color: "007A33" }) ]}));
children.push(moverTable(data.top_gainers, "007A33"));
children.push(new Paragraph({ spacing: { before: 200, after: 80 }, children: [
    new TextRun({ text: "Top Losers", bold: true, font: "Arial", size: 20, color: "CC0000" }) ]}));
children.push(moverTable(data.top_losers, "CC0000"));

// 4. Risk Profile
children.push(secTitle("4. Risk Profile Analysis"));
children.push(body(narrative.risk_analysis));

// 5. Factor Exposures
const fe = data.factor_exposures || {};
const fNames = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"];
const validFactors = fNames.filter(f => fe[f]);
if (validFactors.length > 0) {
    children.push(secTitle("5. Factor Exposures"));
    const fc = [2400, 1740, 1740, 1740, 1740];
    children.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: fc,
        rows: [
            new TableRow({ children: [hCell("Factor", fc[0]), hCell("Exposure", fc[1]), hCell("t-stat", fc[2]), hCell("Significant", fc[3]), hCell("Interpretation", fc[4])] }),
            ...validFactors.map(f => new TableRow({ children: [
                dCell(f, fc[0], { align: AlignmentType.LEFT, bold: true }),
                dCell(fmtN(fe[f].exposure, 4), fc[1]),
                dCell(fmtN(fe[f].t_stat, 2), fc[2]),
                dCell(fe[f].significant ? "Yes" : "No", fc[3]),
                dCell((fe[f].interpretation || "").substring(0, 40), fc[4], { align: AlignmentType.LEFT }),
            ]}))
        ]
    }));
}

// 6. Outlook
children.push(secTitle("6. Outlook & Recommendations"));
children.push(body(narrative.outlook));

// Footer
children.push(new Paragraph({ spacing: { before: 400 }, children: [] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "\u2014 End of Report \u2014", font: "Arial", size: 18, color: "AAAAAA", italics: true })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80 },
    children: [new TextRun({ text: "Generated by Paper Trading Portfolio Dashboard | Data: Yahoo Finance, Kenneth French Data Library",
        font: "Arial", size: 16, color: "AAAAAA" })] }));

const doc = new Document({
    styles: { default: { document: { run: { font: "Arial", size: 20 } } } },
    sections: [{ properties: { page: { size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } }, children }]
});

Packer.toBuffer(doc).then(buf => {
    fs.writeFileSync(outputPath, buf);
    console.log("Report generated: " + outputPath);
}).catch(err => { console.error("Error: " + err.message); process.exit(1); });
