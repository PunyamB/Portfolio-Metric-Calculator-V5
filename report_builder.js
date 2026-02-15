const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, BorderStyle, WidthType, ShadingType } = require("docx");

const args = process.argv.slice(2);
const jsonPath = args[0];
const outputPath = args[1];

const raw = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
const data = raw.data;
const narrative = raw.narrative;
const reportType = raw.report_type;

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function headerCell(text, width) {
    return new TableCell({
        borders, width: { size: width, type: WidthType.DXA },
        shading: { fill: "1B2A4A", type: ShadingType.CLEAR },
        margins: cellMargins,
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
            new TextRun({ text, bold: true, font: "Arial", size: 18, color: "FFFFFF" })
        ]})]
    });
}

function dataCell(text, width, opts = {}) {
    return new TableCell({
        borders, width: { size: width, type: WidthType.DXA },
        shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
        margins: cellMargins,
        children: [new Paragraph({ alignment: opts.align || AlignmentType.CENTER, children: [
            new TextRun({ text: String(text), font: "Arial", size: 18, color: opts.color || "000000", bold: opts.bold || false })
        ]})]
    });
}

function sectionTitle(text) {
    return new Paragraph({
        spacing: { before: 300, after: 120 },
        children: [new TextRun({ text, bold: true, font: "Arial", size: 24, color: "1B2A4A" })]
    });
}

function bodyText(text) {
    return new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun({ text: text || "[No narrative generated]", font: "Arial", size: 20 })]
    });
}

function fmtNum(v, decimals) {
    if (v === null || v === undefined || v === "N/A") return "N/A";
    return Number(v).toFixed(decimals);
}

function fmtDollar(v) {
    if (v === null || v === undefined) return "N/A";
    const n = Number(v);
    const prefix = n < 0 ? "-$" : "$";
    return prefix + Math.abs(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtPct(v) {
    if (v === null || v === undefined || v === "N/A") return "N/A";
    const n = Number(v);
    return (n >= 0 ? "+" : "") + (n * 100).toFixed(2) + "%";
}

function deltaColor(v) {
    if (v === null || v === undefined) return "000000";
    return Number(v) >= 0 ? "007A33" : "CC0000";
}

// --- Build content based on report type ---
let children = [];

if (reportType === "weekly") {
    // Title
    children.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { after: 60 },
        children: [new TextRun({ text: "Weekly Portfolio Review", font: "Arial", size: 40, bold: true, color: "1B2A4A" })]
    }));
    children.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { after: 60 },
        children: [new TextRun({ text: `${data.period_start} \u2014 ${data.period_end}`, font: "Arial", size: 22, color: "666666" })]
    }));
    children.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { after: 300 },
        children: [new TextRun({ text: "Bulls Before Bros | UConn MSFERM FNCE 5322", font: "Arial", size: 20, color: "888888" })]
    }));

    // 1. Executive Summary
    children.push(sectionTitle("1. Executive Summary"));
    children.push(bodyText(narrative.executive_summary));

    // 2. Performance Snapshot
    children.push(sectionTitle("2. Performance Snapshot"));
    children.push(new Paragraph({ spacing: { after: 80 }, children: [] }));

    const snapCols = [2400, 1740, 1740, 1740, 1740];
    const snapHeaders = ["Metric", "Prior Week", "Current Week", "Change", "Signal"];

    const metricsToShow = [
        { label: "Portfolio Value", prior: fmtDollar(data.prior_value), current: fmtDollar(data.current_value), change: fmtDollar(data.value_change), up: data.value_change >= 0 },
        { label: "Positions", prior: String(data.prior_holdings_count), current: String(data.current_holdings_count), change: String(data.current_holdings_count - data.prior_holdings_count), up: true },
    ];

    const metricKeys = ["Sharpe Ratio", "Sortino Ratio", "Beta", "Alpha (Annual)", "VaR (95%)", "Max Drawdown", "Std Dev (Annual)"];
    for (const key of metricKeys) {
        const prior = data.prior_metrics ? data.prior_metrics[key] : null;
        const current = data.current_metrics ? data.current_metrics[key] : null;
        const delta = data.metric_deltas ? data.metric_deltas[key] : null;
        metricsToShow.push({
            label: key,
            prior: prior !== null ? fmtNum(prior, 4) : "N/A",
            current: current !== null ? fmtNum(current, 4) : "N/A",
            change: delta !== null ? (delta >= 0 ? "+" : "") + fmtNum(delta, 4) : "N/A",
            up: delta >= 0,
        });
    }

    const snapRows = metricsToShow.map(m => new TableRow({
        children: [
            dataCell(m.label, snapCols[0], { align: AlignmentType.LEFT, bold: true }),
            dataCell(m.prior, snapCols[1]),
            dataCell(m.current, snapCols[2]),
            dataCell(m.change, snapCols[3], { color: m.up ? "007A33" : "CC0000" }),
            dataCell(m.up ? "\u2191" : "\u2193", snapCols[4], { color: m.up ? "007A33" : "CC0000" }),
        ]
    }));

    children.push(new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: snapCols,
        rows: [
            new TableRow({ children: snapCols.map((w, i) => headerCell(snapHeaders[i], w)) }),
            ...snapRows
        ]
    }));

    children.push(new Paragraph({ spacing: { after: 60 }, children: [
        new TextRun({ text: `Benchmark: S&P 500 weekly return: ${data.benchmark_weekly_return}% | Risk-free rate: ${(data.risk_free_rate * 100).toFixed(2)}%`,
            font: "Arial", size: 16, color: "888888", italics: true })
    ]}));

    // 3. Top Movers
    children.push(sectionTitle("3. Top Movers"));

    const moverCols = [1200, 2600, 1200, 1200, 1200, 1960];
    const moverHeaders = ["Ticker", "Company", "Return %", "$ Impact", "Shares", "Status"];

    function moverTable(movers, titleColor) {
        if (!movers || movers.length === 0) return new Paragraph({ children: [new TextRun({ text: "None", font: "Arial", size: 18, color: "888888" })] });
        return new Table({
            width: { size: 9360, type: WidthType.DXA },
            columnWidths: moverCols,
            rows: [
                new TableRow({ children: moverCols.map((w, i) => headerCell(moverHeaders[i], w)) }),
                ...movers.map(m => new TableRow({
                    children: [
                        dataCell(m.ticker, moverCols[0], { bold: true }),
                        dataCell(m.company || m.ticker, moverCols[1], { align: AlignmentType.LEFT }),
                        dataCell((m.weekly_return_pct >= 0 ? "+" : "") + m.weekly_return_pct + "%", moverCols[2], { color: titleColor }),
                        dataCell(fmtDollar(m.pnl), moverCols[3], { color: titleColor }),
                        dataCell(String(Math.round(m.shares)), moverCols[4]),
                        dataCell(m.note || "", moverCols[5]),
                    ]
                }))
            ]
        });
    }

    children.push(new Paragraph({ spacing: { after: 80 }, children: [
        new TextRun({ text: "Top Gainers", bold: true, font: "Arial", size: 20, color: "007A33" })
    ]}));
    children.push(moverTable(data.top_gainers, "007A33"));

    children.push(new Paragraph({ spacing: { before: 200, after: 80 }, children: [
        new TextRun({ text: "Top Losers", bold: true, font: "Arial", size: 20, color: "CC0000" })
    ]}));
    children.push(moverTable(data.top_losers, "CC0000"));

    children.push(new Paragraph({ spacing: { before: 120 }, children: [] }));
    children.push(bodyText(narrative.movers_commentary));

    // 4. Risk Profile
    children.push(sectionTitle("4. Risk Profile Changes"));
    children.push(bodyText(narrative.risk_analysis));

    // 5. Factor Exposures
    const fe = data.factor_exposures || {};
    if (fe && !fe.error && Object.keys(fe).length > 2) {
        children.push(sectionTitle("5. Factor Exposures"));
        const fCols = [2400, 1740, 1740, 1740, 1740];
        const fHeaders = ["Factor", "Exposure", "t-stat", "Significant", "Interpretation"];
        const factorNames = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"];
        const fRows = factorNames.filter(f => fe[f]).map(f => new TableRow({
            children: [
                dataCell(f, fCols[0], { align: AlignmentType.LEFT, bold: true }),
                dataCell(fmtNum(fe[f].exposure, 4), fCols[1]),
                dataCell(fmtNum(fe[f].t_stat, 2), fCols[2]),
                dataCell(fe[f].significant ? "Yes" : "No", fCols[3]),
                dataCell((fe[f].interpretation || "").substring(0, 40), fCols[4], { align: AlignmentType.LEFT }),
            ]
        }));
        if (fRows.length > 0) {
            children.push(new Table({
                width: { size: 9360, type: WidthType.DXA },
                columnWidths: fCols,
                rows: [new TableRow({ children: fCols.map((w, i) => headerCell(fHeaders[i], w)) }), ...fRows]
            }));
        }
    }

    // 6. Outlook
    children.push(sectionTitle("6. Outlook & Recommendations"));
    children.push(bodyText(narrative.outlook));

} else {
    // INCEPTION REPORT
    children.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { after: 60 },
        children: [new TextRun({ text: "Portfolio Report \u2014 Since Inception", font: "Arial", size: 40, bold: true, color: "1B2A4A" })]
    }));
    children.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { after: 60 },
        children: [new TextRun({ text: `${data.period_start} \u2014 ${data.period_end}`, font: "Arial", size: 22, color: "666666" })]
    }));
    children.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { after: 300 },
        children: [new TextRun({ text: "Bulls Before Bros | UConn MSFERM FNCE 5322", font: "Arial", size: 20, color: "888888" })]
    }));

    // 1. Executive Summary
    children.push(sectionTitle("1. Executive Summary"));
    children.push(bodyText(narrative.executive_summary));

    // 2. Performance Summary
    children.push(sectionTitle("2. Performance Summary"));
    const iCols = [4680, 4680];
    const iData = [
        ["Inception Value", fmtDollar(data.inception_value)],
        ["Current Value", fmtDollar(data.current_value)],
        ["Value Change", fmtDollar(data.value_change)],
        ["Total P&L", fmtDollar(data.total_pnl)],
        ["Unrealised P&L", fmtDollar(data.unrealised_pnl)],
        ["Realised P&L", fmtDollar(data.realised_pnl)],
        ["S&P 500 Return", data.benchmark_return + "%"],
        ["Total Trades", String(data.total_trades)],
        ["Current Positions", String(data.current_holdings_count)],
    ];
    children.push(new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: iCols,
        rows: [
            new TableRow({ children: [headerCell("Metric", iCols[0]), headerCell("Value", iCols[1])] }),
            ...iData.map(r => new TableRow({
                children: [
                    dataCell(r[0], iCols[0], { align: AlignmentType.LEFT, bold: true }),
                    dataCell(r[1], iCols[1]),
                ]
            }))
        ]
    }));

    // Metrics
    children.push(new Paragraph({ spacing: { before: 200 }, children: [] }));
    const mKeys = Object.keys(data.current_metrics || {});
    if (mKeys.length > 0) {
        const mCols = [4680, 4680];
        children.push(new Table({
            width: { size: 9360, type: WidthType.DXA },
            columnWidths: mCols,
            rows: [
                new TableRow({ children: [headerCell("Risk Metric", mCols[0]), headerCell("Current", mCols[1])] }),
                ...mKeys.map(k => new TableRow({
                    children: [
                        dataCell(k, mCols[0], { align: AlignmentType.LEFT, bold: true }),
                        dataCell(fmtNum(data.current_metrics[k], 4), mCols[1]),
                    ]
                }))
            ]
        }));
    }

    // 3. Performance Attribution
    children.push(sectionTitle("3. Performance Attribution"));
    children.push(bodyText(narrative.performance_attribution));

    // Top movers tables
    const moverCols2 = [1200, 2600, 1200, 1200, 1200, 1960];
    const moverHeaders2 = ["Ticker", "Company", "Return %", "$ P&L", "Shares", "Status"];

    function moverTable2(movers, color) {
        if (!movers || movers.length === 0) return new Paragraph({ children: [new TextRun({ text: "None", font: "Arial", size: 18 })] });
        return new Table({
            width: { size: 9360, type: WidthType.DXA },
            columnWidths: moverCols2,
            rows: [
                new TableRow({ children: moverCols2.map((w, i) => headerCell(moverHeaders2[i], w)) }),
                ...movers.map(m => new TableRow({
                    children: [
                        dataCell(m.ticker, moverCols2[0], { bold: true }),
                        dataCell(m.company || m.ticker, moverCols2[1], { align: AlignmentType.LEFT }),
                        dataCell((m.weekly_return_pct >= 0 ? "+" : "") + m.weekly_return_pct + "%", moverCols2[2], { color }),
                        dataCell(fmtDollar(m.pnl), moverCols2[3], { color }),
                        dataCell(String(Math.round(m.shares)), moverCols2[4]),
                        dataCell(m.note || "", moverCols2[5]),
                    ]
                }))
            ]
        });
    }

    children.push(new Paragraph({ spacing: { after: 80 }, children: [
        new TextRun({ text: "Top Performers", bold: true, font: "Arial", size: 20, color: "007A33" })
    ]}));
    children.push(moverTable2(data.top_gainers, "007A33"));
    children.push(new Paragraph({ spacing: { before: 200, after: 80 }, children: [
        new TextRun({ text: "Bottom Performers", bold: true, font: "Arial", size: 20, color: "CC0000" })
    ]}));
    children.push(moverTable2(data.top_losers, "CC0000"));

    // 4. Risk Profile
    children.push(sectionTitle("4. Risk Profile Assessment"));
    children.push(bodyText(narrative.risk_assessment));

    // 5. Factor Exposures
    const fe2 = data.factor_exposures || {};
    if (fe2 && !fe2.error && Object.keys(fe2).length > 2) {
        children.push(sectionTitle("5. Factor Exposures"));
        const fCols2 = [2400, 1740, 1740, 1740, 1740];
        const fHeaders2 = ["Factor", "Exposure", "t-stat", "Significant", "Interpretation"];
        const factorNames2 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"];
        const fRows2 = factorNames2.filter(f => fe2[f]).map(f => new TableRow({
            children: [
                dataCell(f, fCols2[0], { align: AlignmentType.LEFT, bold: true }),
                dataCell(fmtNum(fe2[f].exposure, 4), fCols2[1]),
                dataCell(fmtNum(fe2[f].t_stat, 2), fCols2[2]),
                dataCell(fe2[f].significant ? "Yes" : "No", fCols2[3]),
                dataCell((fe2[f].interpretation || "").substring(0, 40), fCols2[4], { align: AlignmentType.LEFT }),
            ]
        }));
        if (fRows2.length > 0) {
            children.push(new Table({
                width: { size: 9360, type: WidthType.DXA },
                columnWidths: fCols2,
                rows: [new TableRow({ children: fCols2.map((w, i) => headerCell(fHeaders2[i], w)) }), ...fRows2]
            }));
        }
    }

    // 6. Forward Strategy
    children.push(sectionTitle("6. Forward Strategy"));
    children.push(bodyText(narrative.forward_strategy));
}

// Footer
children.push(new Paragraph({ spacing: { before: 400 }, children: [] }));
children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "\u2014 End of Report \u2014", font: "Arial", size: 18, color: "AAAAAA", italics: true })]
}));
children.push(new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 80 },
    children: [new TextRun({ text: "Generated by Paper Trading Portfolio Dashboard | Data: Yahoo Finance, Kenneth French Data Library",
        font: "Arial", size: 16, color: "AAAAAA" })]
}));

const doc = new Document({
    styles: { default: { document: { run: { font: "Arial", size: 20 } } } },
    sections: [{
        properties: {
            page: {
                size: { width: 12240, height: 15840 },
                margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
            }
        },
        children
    }]
});

Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync(outputPath, buffer);
    console.log("Report generated: " + outputPath);
}).catch(err => {
    console.error("Error: " + err.message);
    process.exit(1);
});
